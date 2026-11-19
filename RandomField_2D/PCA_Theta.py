"""Theta, PCA-Theta and eta weighting for multi-output polynomial chaos.

Public entry points
-------------------
ThetaCriterionPCE: UQpy-compatible sampling criterion; decorrelate enables PCA,
    adaptive_weighting enables eta. All four adaptive variants share this class.
svd_loo / choose_degrees: analytical relative squared LOO and degree selection.
run_2d: generic experiment loop used by Gaussian, Wave and RandomField_2D.
select_matrix_batch / run_kl: cached-matrix version for the large 5D example.
save_result / load_result: portable NPZ + JSON result storage.

Example definitions, settings and plotting are in the corresponding notebooks.
Project license: MIT. UQpy attribution is retained in LICENSE_UQpy.txt.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from time import perf_counter
import time, json, hashlib, math
import numpy as np
from scipy.special import eval_legendre
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.linalg import lstsq
from threadpoolctl import threadpool_limits
from beartype import beartype
import UQpy
from UQpy.distributions import JointIndependent, Uniform
from UQpy.surrogates import PolynomialChaosExpansion, polynomial_chaos
from UQpy.surrogates.polynomial_chaos.polynomials.TotalDegreeBasis import TotalDegreeBasis
from UQpy.surrogates.polynomial_chaos.regressions import LeastSquareRegression
MODE = 'svd_squared_loo'



class ThetaCriterionPCE:
    @beartype
    def __init__(
        self,
        surrogates: list[UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion],
        eps: float = 1e-15,
        adaptive_weighting: bool = False,
    ):
        self.surrogates = surrogates
        self.eps = eps
        self.adaptive_weighting = adaptive_weighting
        self.exponent_history = []

    def run(
        self,
        existing_samples: np.ndarray,
        candidate_samples: np.ndarray,
        nsamples: int = 1,
        samples_weights=None,
        candidate_weights=None,
        enable_criterium: bool = False,
        decorrelate: bool = False,
        pca_min_contribution: float = 0.05,
        pca_use_existing_only: bool = True,
        return_pca_info: bool = False,
    ):
        pces = self.surrogates

        nsimexisting, nvar = existing_samples.shape
        nsimcandidate, _ = candidate_samples.shape

        if samples_weights is None:
            samples_weights = np.ones(nsimexisting)

        if candidate_weights is None:
            candidate_weights = np.ones(nsimcandidate)

        samples_weights = np.asarray(samples_weights, dtype=float)
        candidate_weights = np.asarray(candidate_weights, dtype=float)

        pca_info = None

        if decorrelate and len(pces) > 1:
            if pca_use_existing_only:
                X_pca = existing_samples
                W_pca = samples_weights
            else:
                X_pca = np.vstack([existing_samples, candidate_samples])
                W_pca = np.concatenate([samples_weights, candidate_weights])

            pces, pca_info = self._pca_decorrelate_pces(
                pces=pces,
                reference_samples=X_pca,
                reference_weights=W_pca,
                min_contribution=pca_min_contribution,
            )

        pos = []
        criterium_history = []
        candidate_original_indices = np.arange(nsimcandidate)

        # The PCEs remain fixed during batch selection. Standardized
        # coordinates and component-wise local variances are therefore
        # evaluated once and reused throughout the batch.
        S = polynomial_chaos.Polynomials.standardize_sample(
            existing_samples,
            pces[0].polynomial_basis.distributions,
        )

        S_candidate = polynomial_chaos.Polynomials.standardize_sample(
            candidate_samples,
            pces[0].polynomial_basis.distributions,
        )

        variance_existing = np.column_stack([
            self._local_variance(
                coordinates=existing_samples,
                pce=pce,
                weight=samples_weights,
            )
            for pce in pces
        ])

        variance_candidate = np.column_stack([
            self._local_variance(
                coordinates=candidate_samples,
                pce=pce,
                weight=candidate_weights,
            )
            for pce in pces
        ])

        for _ in range(nsamples):
            lengths = self._distance_matrix(S_candidate, S)

            nn_idx = np.argmin(lengths, axis=1)
            l = np.nanmin(lengths, axis=1)

            variance_closest = variance_existing[nn_idx]

            # Normalize candidate and nearest-neighbor contributions
            # separately, as defined over Q_c and Q_s in the paper.
            scale_candidate = np.max(
                variance_candidate,
                axis=0,
            )
            scale_candidate = np.maximum(
                scale_candidate,
                self.eps,
            )

            scale_closest = np.max(
                variance_closest,
                axis=0,
            )
            scale_closest = np.maximum(
                scale_closest,
                self.eps,
            )

            vc = np.sum(
                variance_candidate
                / scale_candidate[None, :],
                axis=1,
            )

            vs = np.sum(
                variance_closest
                / scale_closest[None, :],
                axis=1,
            )

            criterium_v = np.sqrt(
                np.maximum(vc * vs, 0.0)
            )
            criterium_l = np.maximum(l, self.eps) ** nvar

            if self.adaptive_weighting:
                # eta = q_V directly. No square root or artificial lower bound.
                # q_V is computed over unused candidates only. The original
                # component normalizations and Euclidean distances are unchanged.
                qv_started = perf_counter()
                available = l > self.eps
                n_available = int(np.sum(available))
                if n_available == 0:
                    raise RuntimeError('No unused candidate remains.')
                v_available = criterium_v[available]
                sum_v = float(np.sum(v_available))
                sum_v2 = float(np.dot(v_available, v_available))
                zero_variance = sum_v2 == 0.0
                qv = (sum_v * sum_v) / (n_available * sum_v2) if not zero_variance else None
                eta = qv if qv is not None else 0.0
                qv_seconds = perf_counter() - qv_started
                if zero_variance:
                    theta = criterium_l.copy()
                else:
                    if not np.isfinite(qv) or not (0.0 < qv <= 1.0 + 1e-12):
                        raise RuntimeError(f'Invalid q_V: {qv}')
                    theta = np.power(criterium_v, eta) * criterium_l
                theta[~available] = -np.inf

                k = int(np.argmax(theta))
                max_distance = float(np.max(l[available]))
                self.exponent_history.append(dict(
                    N_before_batch=nsimexisting,
                    batch_step=len(pos) + 1,
                    N_before_selection=len(S),
                    candidate_index=int(candidate_original_indices[k]),
                    n_available=n_available, sum_v=sum_v, sum_v2=sum_v2,
                    qv=qv, eta=eta, all_zero_variance=zero_variance,
                    selected_distance=float(l[k]), max_distance=max_distance,
                    relative_distance=float(l[k] / max_distance),
                    qv_statistics_seconds=qv_seconds,
                ))
            else:
                available = l > self.eps
                if not np.any(available):
                    raise RuntimeError('No unused candidate remains.')
                values = criterium_v[available]
                if float(np.dot(values, values)) == 0.0:
                    theta = criterium_l.copy()
                else:
                    theta = criterium_v * criterium_l
                theta[~available] = -np.inf
                k = int(np.argmax(theta))
            pos.append(int(candidate_original_indices[k]))

            if enable_criterium:
                criterium_history.append(theta.copy())

            existing_samples = np.vstack([
                existing_samples,
                candidate_samples[[k], :],
            ])

            samples_weights = np.append(
                samples_weights,
                candidate_weights[k],
            )

            # Temporarily add the selected point to the ED caches so that
            # it can become the nearest neighbor in the next batch step.
            S = np.vstack([
                S,
                S_candidate[[k], :],
            ])

            variance_existing = np.vstack([
                variance_existing,
                variance_candidate[[k], :],
            ])

            candidate_samples = np.delete(
                candidate_samples,
                k,
                axis=0,
            )
            candidate_weights = np.delete(
                candidate_weights,
                k,
                axis=0,
            )
            S_candidate = np.delete(
                S_candidate,
                k,
                axis=0,
            )
            variance_candidate = np.delete(
                variance_candidate,
                k,
                axis=0,
            )
            candidate_original_indices = np.delete(
                candidate_original_indices,
                k,
            )

            if candidate_samples.shape[0] == 0:
                break
        if return_pca_info:
            if enable_criterium:
                return pos, criterium_history, pca_info
            if not enable_criterium and nsamples == 1:
                return pos[0], pca_info
            return pos, pca_info

        if not enable_criterium and nsamples == 1:
            return pos[0]

        if enable_criterium:
            return pos, criterium_history

        return pos

    def _pca_decorrelate_pces(
        self,
        pces,
        reference_samples,
        reference_weights=None,
        min_contribution: float = 0.05,
    ):
        """
        Transform scalar output PCEs into standardized PCA-component PCEs.

        This uses correlation-PCA:

            y_std_j = (y_j - mean_j) / std_j

        rather than covariance-PCA. This avoids domination by outputs with
        larger physical units or amplitudes.
        """

        if reference_samples.shape[0] == 0:
            raise ValueError("reference_samples is empty; PCA decorrelation needs reference samples.")

        if reference_weights is None:
            reference_weights = np.ones(reference_samples.shape[0])

        reference_weights = np.asarray(reference_weights, dtype=float).reshape(-1)
        reference_weights = np.maximum(reference_weights, 0.0)

        if np.sum(reference_weights) <= self.eps:
            reference_weights = np.ones(reference_samples.shape[0])

        reference_weights = reference_weights / np.sum(reference_weights)

        Y = np.column_stack([
            np.asarray(p.predict(reference_samples)).reshape(-1)
            for p in pces
        ])
        # Y = np.column_stack([
        #     np.asarray(p.experimental_design_output).reshape(-1)
        #     for p in pces
        # ])

        y_mean = np.sum(Y * reference_weights[:, None], axis=0)
        Y_centered = Y - y_mean[None, :]

        y_var = np.sum((Y_centered ** 2) * reference_weights[:, None], axis=0)
        y_std = np.sqrt(np.maximum(y_var, 0.0))
        y_std = np.maximum(y_std, self.eps)

        Y_standardized = Y_centered / y_std[None, :]

        C = (Y_standardized * reference_weights[:, None]).T @ Y_standardized
        C = 0.5 * (C + C.T)

        eigvals, eigvecs = np.linalg.eigh(C)

        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]

        total = np.sum(eigvals)

        if total <= self.eps:
            contribution = np.zeros_like(eigvals)
            keep = np.zeros_like(eigvals, dtype=bool)
            keep[0] = True
        else:
            contribution = eigvals / total
            keep = contribution >= min_contribution

            if not np.any(keep):
                keep[0] = True

        eigvals_kept = eigvals[keep]
        eigvecs_kept = eigvecs[:, keep]
        contribution_kept = contribution[keep]

        ref_basis = self._build_common_total_degree_basis(pces)
        ref_mi = ref_basis.multi_index_set

        # Nested total-degree bases use exact index copying/zero padding.
        # No Gram inverse is needed on this path.
        A_inv = None

        coeffs_full = []
        biases = []

        for p in pces:
            c_full, b = self._project_pce_to_basis(
                pce=p,
                ref_basis=ref_basis,
                ref_multi_index=ref_mi,
                reference_samples=reference_samples,
                A_inv=A_inv,
            )

            coeffs_full.append(c_full)
            biases.append(b)

        B = np.column_stack(coeffs_full)
        b_vec = np.asarray(biases, dtype=float)

        B_std = B / y_std[None, :]
        b_std = (b_vec - y_mean) / y_std

        B_new = B_std @ eigvecs_kept
        b_new = b_std @ eigvecs_kept

        transformed = []

        for i in range(B_new.shape[1]):
            p_new = pces[0].__class__(
                polynomial_basis=ref_basis,
                regression_method=pces[0].regression_method,
            )

            p_new.coefficients = B_new[:, i]
            p_new.bias = float(b_new[i])

            if hasattr(p_new, "outputs_number"):
                p_new.outputs_number = 1

            if hasattr(p_new, "design_matrix"):
                p_new.design_matrix = None

            transformed.append(p_new)

        pca_info = {
            "eigenvalues": eigvals,
            "contribution": contribution,
            "kept_mask": keep,
            "kept_indices": np.where(keep)[0],
            "kept_eigenvalues": eigvals_kept,
            "kept_contribution": contribution_kept,
            "n_original_outputs": len(pces),
            "n_kept_outputs": len(transformed),
            "min_contribution": min_contribution,
            "mean": y_mean,
            "std": y_std,
            "components": eigvecs_kept,
            "pca_type": "standardized",
        }

        return transformed, pca_info

    @staticmethod
    def _build_common_total_degree_basis(pces):
        degrees = []

        for p in pces:
            mi = p.polynomial_basis.multi_index_set
            degrees.append(int(np.sum(mi, axis=1).max()))

        ref_degree = max(degrees)
        distributions = pces[0].polynomial_basis.distributions

        return TotalDegreeBasis(distributions, ref_degree)

    def _project_pce_to_basis(
        self,
        pce,
        ref_basis,
        ref_multi_index,
        reference_samples,
        A_inv,
    ):
        sub_mi = pce.polynomial_basis.multi_index_set
        beta = np.asarray(pce.coefficients)

        if beta.ndim > 1:
            beta = beta.reshape(-1)

        b = float(pce.bias) if getattr(pce, "bias", None) is not None else 0.0

        ref_map = {
            tuple(row.tolist()): i
            for i, row in enumerate(ref_multi_index)
        }

        pos = np.array([
            ref_map.get(tuple(row.tolist()), -1)
            for row in sub_mi
        ], dtype=int)

        can_copy = (
            beta.shape[0] == sub_mi.shape[0]
            and np.all(pos >= 0)
        )

        if can_copy:
            c_full = np.zeros(ref_multi_index.shape[0])
            c_full[pos] = beta
            return c_full, b

        if A_inv is None:
            raise RuntimeError("Non-nested basis cannot use the exact coefficient lifting path.")
        y = np.asarray(pce.predict(reference_samples)).reshape(-1)
        y_without_bias = y - b

        Phi_ref = ref_basis.evaluate_basis(reference_samples)
        c_full = A_inv @ (Phi_ref.T @ y_without_bias)

        return c_full, b

    @staticmethod
    def _centered_pce_response(coordinates, pce, weight=1.0):
        beta = np.asarray(pce.coefficients)

        if beta.ndim == 1:
            beta_nc = beta.copy()
            beta_nc[0] = 0.0
        else:
            beta_nc = beta.copy()
            beta_nc[0, :] = 0.0

        Phi = pce.polynomial_basis.evaluate_basis(coordinates)

        if np.isscalar(weight):
            Phi_w = Phi * float(weight)
        else:
            Phi_w = Phi * np.asarray(weight, dtype=float)[:, None]

        Y = Phi_w @ beta_nc

        if Y.ndim > 1 and Y.shape[1] == 1:
            Y = Y[:, 0]

        return np.asarray(Y).reshape(coordinates.shape[0], -1)

    def _centered_pce_responses(self, coordinates, pces, weight=1.0):
        G_list = []

        for pce in pces:
            g = self._centered_pce_response(
                coordinates=coordinates,
                pce=pce,
                weight=weight,
            )

            if g.shape[1] > 1:
                g = np.sum(g, axis=1, keepdims=True)

            G_list.append(g[:, 0])

        return np.column_stack(G_list)

    def _local_covariance_matrices(self, coordinates, pces, weight=1.0):
        G = self._centered_pce_responses(
            coordinates=coordinates,
            pces=pces,
            weight=weight,
        )

        C_loc = np.einsum("ki,kj->kij", G, G)

        pdf = polynomial_chaos.Polynomials.standardize_pdf(
            coordinates,
            pces[0].polynomial_basis.distributions,
        )

        pdf = np.asarray(pdf, dtype=float).reshape(-1)

        C_loc = C_loc * pdf[:, None, None]
        C_loc = 0.5 * (C_loc + np.transpose(C_loc, (0, 2, 1)))

        return C_loc

    def _redundancy_adjusted_effective_variance(
        self,
        C_norm: np.ndarray,
        R_global: np.ndarray,
    ):
        R_abs = np.abs(R_global)

        T = np.trace(C_norm, axis1=1, axis2=2)

        D = np.sum(
            R_abs[None, :, :] * np.abs(C_norm),
            axis=(1, 2),
        )

        V_eff = (T ** 2) / np.maximum(D, self.eps)

        V_eff = np.nan_to_num(
            V_eff,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return np.maximum(V_eff, 0.0)

    @staticmethod
    def _safe_corrcoef(Y):
        Y = np.asarray(Y, dtype=float)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n_outputs = Y.shape[1]

        if n_outputs == 1:
            return np.ones((1, 1))

        R = np.corrcoef(Y, rowvar=False)

        R = np.nan_to_num(
            R,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        R = 0.5 * (R + R.T)

        np.fill_diagonal(R, 1.0)

        return R

    @staticmethod
    def _distance_matrix(candidates, existing):
        return cdist(candidates, existing)

    @staticmethod
    def _local_variance(coordinates, pce, weight=1):
        beta = np.asarray(pce.coefficients)

        if beta.ndim == 1:
            beta_nc = beta.copy()
            beta_nc[0] = 0.0
        else:
            beta_nc = beta.copy()
            beta_nc[0, :] = 0.0

        Phi = pce.polynomial_basis.evaluate_basis(coordinates)

        if np.isscalar(weight):
            Phi_w = Phi * float(weight)
        else:
            Phi_w = Phi * np.asarray(weight)[:, None]

        Y = Phi_w @ beta_nc

        if Y.ndim == 1:
            var_local = Y ** 2
        else:
            var_local = np.sum(Y ** 2, axis=1)

        pdf = polynomial_chaos.Polynomials.standardize_pdf(
            coordinates,
            pce.polynomial_basis.distributions,
        )

        return var_local * pdf



# OLS fitting and experimental designs


def make_polynomial_basis(bounds, degree):
    marginals = [
        Uniform(loc=float(lower), scale=float(upper - lower))
        for lower, upper in bounds
    ]
    return TotalDegreeBasis(JointIndependent(marginals=marginals), int(degree))

def latin_hypercube(seed, n_samples, bounds):
    rng = np.random.default_rng(seed)
    samples = np.empty((n_samples, len(bounds)), dtype=float)
    for j, (lower, upper) in enumerate(bounds):
        permutation = rng.permutation(n_samples)
        unit = (permutation + rng.random(n_samples)) / n_samples
        samples[:, j] = lower + (upper - lower) * unit
    return samples

def random_samples(seed, n_samples, bounds):
    rng = np.random.default_rng(seed)
    unit = rng.random((n_samples, len(bounds)))
    lower = np.asarray([item[0] for item in bounds], dtype=float)
    upper = np.asarray([item[1] for item in bounds], dtype=float)
    return lower + unit * (upper - lower)

def candidate_grid(bounds, n_grid_1d):
    axes = [np.linspace(low, high, n_grid_1d) for low, high in bounds]
    meshes = np.meshgrid(*axes, indexing="xy")
    return np.column_stack([mesh.ravel() for mesh in meshes])

def svd_loo(a,y,return_pointwise=False):
    n,P=a.shape; q=y.shape[1]; eps=np.finfo(float).eps
    row=dict(P=P,reason='',loo=np.full(q,np.inf),coef=None,condition=np.inf,min_den=np.nan)
    if n<P:row['reason']='N<P';return row
    if n==P:row['reason']='square_PRESS_undefined';return row
    # Same OLS solve as UQpy. No Gram inverse, ridge penalty, or truncated-SVD fit.
    coef=np.linalg.lstsq(a,y,rcond=None)[0]
    u,s,_=np.linalg.svd(a,full_matrices=False)
    den=1-np.sum(u*u,axis=1);rank_tol=max(a.shape)*eps*s[0];guard=10*max(a.shape)*eps
    row.update(coef=coef,condition=float(s[0]/s[-1]),min_den=float(den.min()),
               min_den_index=int(np.argmin(den)),rank=int(np.sum(s>rank_tol)),guard=float(guard))
    if s[-1]<=rank_tol:row['reason']='rank';return row
    if n>P and np.any(den<=guard):row['reason']='leverage';return row
    var=np.var(y,axis=0,ddof=1)
    if np.any(var<=0):row['reason']='variance';return row
    residual=y-a@coef
    normalized=(residual/den[:,None])**2/var
    row.update(loo_source='svd_fast',svd_loo_defined=True)
    loo=normalized.mean(axis=0);total=normalized.sum(axis=0)
    order=np.argsort(normalized,axis=0)
    topidx=np.argmax(normalized,axis=0); topvalues=np.max(normalized,axis=0)
    safe=np.maximum(total,np.finfo(float).tiny)
    row.update(loo=loo,train_nmse=np.mean(residual**2,axis=0)/var,
        top1=topvalues/safe,top5=np.sort(normalized,axis=0)[-5:].sum(axis=0)/safe,
        top_index=topidx,den_at_top=den[topidx],residual_at_top=residual[topidx,np.arange(q)],
        score=float(loo.max()),worst_output=int(loo.argmax()),
        max_hat=float(1-den.min()))
    row['standard_loo']=loo.copy()
    row['gcv']=row['train_nmse']/(1-P/n)**2
    row['scoring_rule']='gcv' if MODE=='gcv' else 'squared_LOO'
    if MODE=='gcv':
        row.update(loo=row['gcv'].copy(),score=float(row['gcv'].max()),worst_output=int(row['gcv'].argmax()),loo_source='gcv')
    if return_pointwise:row['point_losses']=normalized
    if not np.all(np.isfinite(loo)):row['reason']='nonfinite'
    return row

def choose_degrees(fits,mode,tolerance):
    valid=[p for p,f in fits.items() if not f['reason']]
    if not valid:raise RuntimeError('No admissible degree')
    scores=np.stack([fits[p]['loo'] for p in valid])
    if mode=='common':scores=scores.max(axis=1,keepdims=True)
    selected=[]
    for j in range(scores.shape[1]):
        acceptable=np.flatnonzero(scores[:,j]<=tolerance)
        selected.append(valid[int(acceptable[0])] if len(acceptable) else valid[int(np.argmin(scores[:,j]))])
    if mode=='common':selected*=len(fits[valid[0]]['loo'])
    return np.array(selected,dtype=int)


# Matrix operations for the cached 5D benchmark


def total_degree_indices(d,p):
    def exact(total, dim):
        if dim == 1:
            yield (total,)
        else:
            for k in range(total+1):
                for tail in exact(total-k,dim-1): yield (k,)+tail
    return np.asarray([a for total in range(p+1) for a in exact(total,d)],dtype=int)

def legendre_basis(X, mi):
    X=np.asarray(X); p=int(mi.max()); Z=2*X-1
    tables=[np.column_stack([np.sqrt(2*k+1)*eval_legendre(k,Z[:,j]) for k in range(p+1)]) for j in range(X.shape[1])]
    A=np.ones((len(X),len(mi)))
    for j,t in enumerate(tables): A *= t[:,mi[:,j]]
    return A

class Timer:
    def __init__(self): self.wall=defaultdict(float); self.cpu=defaultdict(float); self.eta_history=[]
    @contextmanager
    def measure(self,key):
        t=time.perf_counter(); c=time.process_time()
        try: yield
        finally:
            self.wall[key]+=time.perf_counter()-t; self.cpu[key]+=time.process_time()-c

def matrix_errors(pred,true):
    scale=np.maximum(np.sqrt(np.mean(true**2,axis=0)),1e-15)
    a=np.abs(pred-true)/scale
    per=np.sqrt(np.mean(a*a,axis=0))
    return dict(nrmse=float(per.mean()),nmae=float(a.mean()),nmaxae=float(a.max()),nrmse_outputs=per)

def fit_matrix_pce(A,Y,degrees,d,timer):
    # Factor once per distinct degree and solve all associated outputs together.
    # Both Theta and PCA-Theta use exactly this construction, with no PCA refit.
    B=np.zeros((A.shape[1],Y.shape[1]))
    with timer.measure('ols_fitting'):
        for p in sorted(set(degrees)):
            cols=np.flatnonzero(np.asarray(degrees)==p); P=math.comb(d+p,d)
            beta,_,rank,_=lstsq(A[:,:P],Y[:,cols],cond=None,lapack_driver='gelsy')
            if rank<P: raise RuntimeError(f'Rank-deficient OLS: rank {rank}, basis {P}.')
            B[:P,cols]=beta
    return B

def pca_coefficients(A,B,threshold,timer,Y_ed=None):
    with timer.measure('pca_processing'):
        Y=A@B
        # Y=np.asarray(Y_ed,dtype=float)
        mean=Y.mean(axis=0); std=Y.std(axis=0,ddof=1)
        std=np.maximum(std,1e-15)
        Z=(Y-mean)/std; C=Z.T@Z/(len(Y)-1)
        w,V=np.linalg.eigh(C); order=np.argsort(w)[::-1]
        w=np.maximum(w[order],0); V=V[:,order]
        r=w/max(w.sum(),1e-15); keep=r>=threshold
        if not keep.any(): keep[0]=True
        Bs=B/std; Bs=Bs.copy(); Bs[0]-=mean/std
        rotated=Bs@V[:,keep]
    return rotated,dict(contribution=r,kept=np.flatnonzero(keep),n_kept=int(keep.sum()))

def local_values(A,B,timer,chunk=4096):
    # Legendre basis is shared and cached for ALL methods. No repeated basis
    # evaluation per output is deliberately introduced to inflate PCA speedup.
    with timer.measure('criterion_response_evaluation'):
        out=np.empty((A.shape[0],B.shape[1]))
        for lo in range(0,len(A),chunk):
            z=A[lo:lo+chunk,1:]@B[1:]
            out[lo:lo+chunk]=z*z
    return out

def select_matrix_batch(X,Xc,Ve,Vc,active,count,d,timer,adaptive_weighting=False):
    # Standardized uniform coordinates are 2*x-1; global factor cancels ranking.
    with timer.measure('distance_search'):
        tree=cKDTree(2*X-1)
        dist,nn=tree.query(2*Xc-1,workers=1)
    chosen=[]; ve=Ve.copy()
    for _ in range(count):
        with timer.measure('criterion_aggregation'):
            ids=np.flatnonzero(active)
            near=ve[nn[ids]]; cand=Vc[ids]
            cs=np.maximum(cand.max(axis=0),1e-15)
            ss=np.maximum(near.max(axis=0),1e-15)
            variance = np.sqrt((cand/cs).sum(axis=1) * (near/ss).sum(axis=1))
            if adaptive_weighting:
                available = dist[ids] > 1e-14
                values = variance[available]
                if not values.size:
                    raise RuntimeError('No unused candidate remains.')
                squared_sum = float(values @ values)
                if squared_sum:
                    eta = float(values.sum())**2 / (len(values) * squared_sum)
                    if not np.isfinite(eta) or not 1/len(values)-1e-14 <= eta <= 1+1e-12:
                        raise RuntimeError(f'Invalid adaptive exponent: {eta}')
                    score = variance**eta * dist[ids]**d
                else:
                    eta = 0.
                    score = dist[ids]**d
                timer.eta_history.append(eta)
                score[~available]=-np.inf
            else:
                available = dist[ids] > 1e-14
                values = variance[available]
                if not values.size:
                    raise RuntimeError('No unused candidate remains.')
                if float(values @ values) == 0.0:
                    score = dist[ids]**d
                else:
                    score = variance * dist[ids]**d
                score[~available] = -np.inf
            if not np.isfinite(score).any(): raise RuntimeError('No unused candidate remains.')
            k=int(ids[np.argmax(score)]); chosen.append(k); active[k]=False
            ve=np.vstack([ve,Vc[k]])
        with timer.measure('distance_search'):
            newdist=2*np.linalg.norm(Xc-Xc[k],axis=1)
            closer=newdist<dist
            dist[closer]=newdist[closer]; nn[closer]=len(ve)-1
    return np.asarray(chosen)

# Generic experiment helpers, appended to the single public algorithm module.

METHODS = ('random_ols', 'lhs_ols', 'theta_ols', 'pca_theta_ols',
           'eta_theta_ols', 'eta_pca_theta_ols')


def evaluate_errors(prediction, truth):
    """Output-wise errors, normalized by the fixed test-response RMS."""
    delta = prediction - truth
    absolute = np.abs(delta)
    scale = np.maximum(np.sqrt(np.mean(truth**2, axis=0)), np.finfo(float).tiny)
    values = {
        'nrmse_outputs': np.sqrt(np.mean(delta**2, axis=0)) / scale,
        'nmae_outputs': absolute.mean(axis=0) / scale,
        'nmaxae_outputs': absolute.max(axis=0) / scale,
    }
    values.update(nrmse=float(values['nrmse_outputs'].mean()),
                  nmae=float(values['nmae_outputs'].mean()),
                  nmaxae=float(values['nmaxae_outputs'].max()))
    return values


class BasisCache:
    """One Legendre matrix per input set, reused for every output and degree.

    The UQpy basis evaluator is restored after each experiment. Run concurrent
    experiments in separate processes, not threads, because UQpy's evaluator
    is temporarily shared while a cache is installed.
    """

    def __init__(self, bounds, p_min=5, p_max=25):
        self.bounds = bounds
        self.p_max = p_max
        self.bases = {p: make_polynomial_basis(bounds, p) for p in range(p_min, p_max+1)}
        self.mi = self.bases[p_max].multi_index_set.astype(int)
        lookup = {tuple(row): i for i, row in enumerate(self.mi)}
        self.indices = {p: np.array([lookup[tuple(row)] for row in b.multi_index_set])
                        for p, b in self.bases.items()}
        self.cache = OrderedDict()
        self.basis_type = type(self.bases[p_max])

    def matrix(self, x):
        x = np.asarray(x, dtype=float)
        key = (x.shape, hashlib.blake2b(x.tobytes(), digest_size=16).digest())
        if key not in self.cache:
            lo, hi = np.asarray(self.bounds).T
            z = 2*(x-lo)/(hi-lo)-1
            tables = [np.column_stack([eval_legendre(k, z[:, j])*np.sqrt(2*k+1)
                                       for k in range(self.p_max+1)]) for j in range(x.shape[1])]
            # This expression preserves the two-dimensional paper implementation.
            self.cache[key] = tables[0][:, self.mi[:, 0]]*tables[1][:, self.mi[:, 1]]
            if len(self.cache) > 5:
                self.cache.popitem(last=False)
        return self.cache[key]

    @contextmanager
    def installed(self):
        original = self.basis_type.evaluate_basis
        lookup = {tuple(row): i for i, row in enumerate(self.mi)}

        def evaluate(basis, x):
            ids = np.array([lookup[tuple(row)] for row in basis.multi_index_set])
            return self.matrix(x)[:, ids]

        self.basis_type.evaluate_basis = evaluate
        try:
            yield self
        finally:
            self.basis_type.evaluate_basis = original

    def surrogates(self, x, y, degrees, fits):
        models = []
        for j, degree in enumerate(degrees):
            basis = self.bases[int(degree)]
            model = PolynomialChaosExpansion(polynomial_basis=basis,
                                              regression_method=LeastSquareRegression())
            model.coefficients = fits[int(degree)]['coef'][:, j:j+1]
            model.bias = None
            model.experimental_design_input = x
            model.experimental_design_output = y[:, j]
            model.design_matrix = basis.evaluate_basis(x)
            model.outputs_number = 1
            models.append(model)
        return models


def run_2d(model, bounds, *, method='eta_pca_theta_ols', degree_mode='common',
           degree_rule='loo', repeat=0, n_init=30, n_final=1000, batch_size=10,
           p_min=5, p_max=25, n_grid=100, n_test=1000, test_seed=2026,
           seed=None, pca_threshold=.01, progress=None):
    """One paired 2D trajectory; model and settings are supplied by a notebook.

    degree_rule='max' chooses the same maximum admissible degree for all
    methods. degree_rule='loo' uses the analytical SVD-based LOO score.
    LHS is a complete independent design at each budget beyond the shared
    initial random ED. Test responses are used only for reporting errors.
    """
    method = method.replace('qv_', 'eta_')
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    if degree_mode not in ('common', 'individual') or degree_rule not in ('loo', 'max'):
        raise ValueError('Invalid degree-selection mode or rule')
    if len(bounds) != 2 or n_final < n_init or batch_size < 1:
        raise ValueError('run_2d requires two inputs and a valid sample budget')
    seed = 123+1000*repeat if seed is None else int(seed)
    weighted, decorrelate = method.startswith('eta_'), 'pca' in method
    candidates = candidate_grid(bounds, n_grid)
    xt = random_samples(test_seed, n_test, bounds)
    yt = model(xt)
    x = random_samples(seed, n_init, bounds)
    y = model(x)
    x_initial = x.copy()
    random_generator = np.random.default_rng(seed+777)
    bank = BasisCache(bounds, p_min, p_max)
    timer = Timer()
    history, pca_history, exponent_history, diagnostics, designs = [], [], [], [], {}
    detailed = defaultdict(float)

    class TrackedCriterion(ThetaCriterionPCE):
        def _measure(self, key, function, *args, **kwargs):
            start = time.process_time()
            try:
                return function(*args, **kwargs)
            finally:
                detailed[key] += time.process_time()-start

        def _distance_matrix(self, *args):
            return self._measure('distance_search', super()._distance_matrix, *args)

        def _local_variance(self, *args, **kwargs):
            return self._measure('criterion_response_evaluation', super()._local_variance, *args, **kwargs)

        def _pca_decorrelate_pces(self, *args, **kwargs):
            return self._measure('pca_processing', super()._pca_decorrelate_pces, *args, **kwargs)

    start_cpu, start_wall = time.process_time(), time.perf_counter()
    with threadpool_limits(limits=1), bank.installed():
        n = n_init
        while True:
            if method == 'lhs_ols':
                if n > n_init:
                    with timer.measure('design_generation'):
                        x = latin_hypercube(np.random.SeedSequence([seed, 888, n]), n, bounds)
                    with timer.measure('model_evaluation'):
                        y = model(x)
                designs[n] = x.copy()
            with timer.measure('degree_selection_and_ols'):
                a = bank.matrix(x)
                if degree_rule == 'max':
                    degree = max(p for p in bank.bases if n > len(bank.indices[p]))
                    coef, _, rank, singular = np.linalg.lstsq(a[:, bank.indices[degree]], y, rcond=None)
                    fits = {degree: dict(coef=coef, loo=np.full(y.shape[1], np.nan), reason='',
                                         P=len(bank.indices[degree]), rank=int(rank))}
                    degrees = np.full(y.shape[1], degree, dtype=int)
                else:
                    fits = {p: svd_loo(a[:, ids], y) for p, ids in bank.indices.items()}
                    degrees = choose_degrees(fits, degree_mode, 0.)
                pces = bank.surrogates(x, y, degrees, fits)
                selected_loo = np.array([fits[int(p)]['loo'][j] for j, p in enumerate(degrees)])
            with timer.measure('test_error_evaluation'):
                predicted = np.column_stack([p.predict(xt).reshape(-1) for p in pces])
                errors = evaluate_errors(predicted, yt)
            history.append(dict(n=n, p_selected=int(degrees.max()), p_selected_outputs=degrees.copy(),
                                selected_loo=selected_loo, **errors))
            diagnostics.append({int(p): {k: f[k] for k in ('P','reason','loo','condition','min_den','rank','guard')
                                         if k in f} for p, f in fits.items()})
            if progress:
                progress(dict(method=method, repeat=repeat, n=n, p=degrees.copy(), **errors))
            if n == n_final:
                break
            count = min(batch_size, n_final-n)
            if method == 'lhs_ols':
                n += count
                continue
            if method == 'random_ols':
                lo, hi = np.asarray(bounds).T
                xn = lo+random_generator.random((count, 2))*(hi-lo)
            else:
                criterion = TrackedCriterion(pces, adaptive_weighting=weighted)
                with timer.measure('theta_batch_selection'):
                    selected, _, info = criterion.run(x, candidates, nsamples=count,
                        enable_criterium=True, decorrelate=decorrelate,
                        pca_min_contribution=pca_threshold, return_pca_info=True)
                xn = candidates[np.atleast_1d(selected).astype(int)]
                exponent_history.extend(criterion.exponent_history)
                if info is not None:
                    pca_history.append(dict(n=n, **info))
            with timer.measure('model_evaluation'):
                yn = model(xn)
            with timer.measure('data_update'):
                x, y = np.vstack([x, xn]), np.vstack([y, yn])
            n = len(x)
        total_cpu, total_wall = time.process_time()-start_cpu, time.perf_counter()-start_wall
        # Like the paper, final PCA plotting data are computed outside construction timing.
        pca_final = None
        if decorrelate:
            _, pca_final = ThetaCriterionPCE(pces)._pca_decorrelate_pces(pces, x, min_contribution=pca_threshold)
    cpu = dict(timer.cpu)
    cpu.update(detailed)
    cpu['criterion_aggregation'] = max(0., cpu.get('theta_batch_selection',0.)-sum(detailed.values()))
    cpu['construction_total'] = total_cpu-cpu.get('test_error_evaluation',0.)
    hist = {k: np.asarray([stage[k] for stage in history]) for k in history[0]}
    return dict(complete=True, method=method, degree_mode=degree_mode, degree_rule=degree_rule,
                repeat=repeat, seed=seed, hist=hist, diagnostics=diagnostics,
                X_initial=x_initial, X_train_final=x, Y_train_final=y, X_train_by_n=designs,
                X_test=xt, Y_test_true=yt, Y_test_pred_final=predicted,
                final_coefficients=[p.coefficients.reshape(-1) for p in pces],
                final_multi_indices=[p.polynomial_basis.multi_index_set for p in pces],
                pca_infos=pca_history, pca_final_info=pca_final, exponent_history=exponent_history,
                cpu_timing=cpu, total_cpu=total_cpu, total_wall=total_wall)


def save_result(path, value):
    """Save arrays and JSON metadata in one NPZ; no executable pickle payload."""
    path = Path(path)
    if path.suffix != '.npz':
        raise ValueError('Result filename must end with .npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}

    def pack(v):
        if isinstance(v, np.ndarray):
            if v.dtype.kind == 'O':
                return pack(v.tolist())
            name = f'array_{len(arrays):05d}'
            arrays[name] = v
            return {'__array__': name}
        if isinstance(v, np.generic):
            return pack(v.item())
        if isinstance(v, dict):
            return {str(k): pack(x) for k, x in v.items()}
        if isinstance(v, (tuple, list)):
            return [pack(x) for x in v]
        if isinstance(v, float) and not np.isfinite(v):
            return {'__float__': str(v)}
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        raise TypeError(f'Unsupported result value: {type(v)}')

    arrays['__metadata__'] = np.asarray(json.dumps(pack(value)))
    temporary = path.with_suffix('.partial.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_result(path):
    """Read a result produced by save_result, with allow_pickle=False."""
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z['__metadata__']))

        def unpack(v):
            if isinstance(v, dict):
                if set(v) == {'__array__'}:
                    return z[v['__array__']].copy()
                if set(v) == {'__float__'}:
                    return float(v['__float__'])
                return {k: unpack(x) for k, x in v.items()}
            if isinstance(v, list):
                return [unpack(x) for x in v]
            return v

        return unpack(metadata)


def prepare_matrix_cache(folder, inputs, degree):
    """Create disposable 5D Legendre matrices, identical for every method."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    indices = total_degree_indices(inputs['initial_X'].shape[1], degree)
    signature = hashlib.sha256(str(degree).encode())
    for key in ('initial_X','test_X','candidates'):
        signature.update(np.ascontiguousarray(inputs[key]).tobytes())
    stamp = folder/'cache.json'
    fingerprint = signature.hexdigest()
    if stamp.exists() and json.loads(stamp.read_text())['fingerprint'] != fingerprint:
        raise RuntimeError('Cache settings differ. Choose a new cache directory.')
    for name, x in [('initial',inputs['initial_X']),('test',inputs['test_X']),('candidate',inputs['candidates'])]:
        path = folder/(name+'_basis.npy')
        if not stamp.exists() or not path.exists():
            temporary = folder/(name+'_basis.partial.npy')
            a = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float64, shape=(len(x),len(indices)))
            for lo in range(0,len(x),2048):
                a[lo:lo+2048] = legendre_basis(x[lo:lo+2048],indices)
            a.flush()
            del a
            temporary.replace(path)
    stamp.write_text(json.dumps(dict(fingerprint=fingerprint)),encoding='utf-8')
    return tuple(np.load(folder/(name+'_basis.npy'),mmap_mode='r') for name in ('initial','test','candidate'))


def run_kl(model, inputs, matrices, config, method, output_path, metadata=None, progress=None):
    """Fixed-degree KL construction with checkpoint recovery and CPU timing.

    The paper uses N=3010..8000, p=10 and batches of ten. The model, field
    realization, candidate inputs and all physical settings live in the notebook.
    """
    if method not in ('theta','pca_theta','eta_theta','eta_pca_theta'):
        raise ValueError(method)
    weighted = method.startswith('eta_')
    output_path = Path(output_path)
    checkpoint = output_path.with_name(output_path.stem+'_checkpoint.npz')
    signature = hashlib.sha256(json.dumps(config,sort_keys=True).encode())
    for key in ('initial_X','initial_Y','test_X','test_Y','candidates'):
        signature.update(np.ascontiguousarray(inputs[key]).tobytes())
    signature = signature.hexdigest()
    if output_path.exists():
        result = load_result(output_path)
        if not result['complete'] or result['fingerprint'] != signature:
            raise RuntimeError('Saved result does not match these settings.')
        return result
    timer = Timer()
    d, degree, batch, nf = (config[k] for k in ('dimension','p','batch_size','n_final'))
    degrees = [degree]*config['n_outputs']
    ai, at, ac = matrices
    x, y, a = inputs['initial_X'].copy(), inputs['initial_Y'].copy(), ai.copy()
    xc, yt = inputs['candidates'], inputs['test_Y']
    active = np.ones(len(xc),dtype=bool)
    hist, pcs, ids = [], [], []
    elapsed_cpu = elapsed_wall = 0.
    if checkpoint.exists():
        state = load_result(checkpoint)
        if state['fingerprint'] != signature:
            raise RuntimeError('Checkpoint does not match these settings.')
        ids, hist, pcs = state['selected'], state['hist'], state['pca']
        x = np.vstack([x,xc[ids]])
        y = state['Y']
        a = np.vstack([a,ac[ids]])
        active[ids] = False
        timer.cpu.update(state['cpu_timing']); timer.wall.update(state['timing'])
        timer.eta_history = state['eta_history']
        elapsed_cpu, elapsed_wall = state['total_cpu'], state['total_time']
    # Warm-up is outside the measured construction, as in the paper run.
    float(np.sum(ac))
    with threadpool_limits(limits=1):
        while True:
            n = len(x)
            start_cpu, start_wall = time.process_time(), time.perf_counter()
            b = fit_matrix_pce(a,y,degrees,d,timer)
            with timer.measure('test_error_evaluation'):
                errors = matrix_errors(at@b,yt)
            hist.append(dict(n=n,**errors))
            if n < nf:
                if 'pca' in method:
                    rotated, info = pca_coefficients(a,b,config['pca_threshold'],timer,Y_ed=y)
                    pcs.append(dict(n=n,**info))
                else:
                    rotated = b
                ve = local_values(a,rotated,timer)
                vc = local_values(ac,rotated,timer)
                new_ids = select_matrix_batch(x,xc,ve,vc,active,min(batch,nf-n),d,timer,
                                             adaptive_weighting=weighted)
                with timer.measure('model_evaluation'):
                    yn = model(xc[new_ids])
                with timer.measure('data_update'):
                    x=np.vstack([x,xc[new_ids]]); y=np.vstack([y,yn]); a=np.vstack([a,ac[new_ids]])
                    ids.extend(new_ids.tolist())
            elapsed_cpu += time.process_time()-start_cpu
            elapsed_wall += time.perf_counter()-start_wall
            state=dict(complete=n==nf,method=method,fingerprint=signature,selected=ids,hist=hist,pca=pcs,
                       Y=y,timing=dict(timer.wall),cpu_timing=dict(timer.cpu),eta_history=timer.eta_history,
                       total_cpu=elapsed_cpu,total_time=elapsed_wall)
            if progress:
                progress(dict(method=method,n=n,**errors))
            if n==nf:
                break
            save_result(checkpoint,state)
    state.update(config=config,degrees=degrees,X=x,B=b,indices=total_degree_indices(d,degree),metadata=metadata)
    save_result(output_path,state)
    return state

