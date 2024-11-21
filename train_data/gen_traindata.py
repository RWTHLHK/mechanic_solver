import numpy as np
from scipy.optimize import root

def assemble_elasticity_tensor(lambda_modulus, mu_modulus):
    """Assemble the 3D elastic stiffness tensor in Voigt notation."""
    # Initialize a 6x6 matrix
    Ce = np.zeros((6, 6))
    
    # Diagonal terms: lambda + 2 * mu
    diagonal_term = lambda_modulus + 2 * mu_modulus
    Ce[0, 0] = Ce[1, 1] = Ce[2, 2] = diagonal_term
    
    # Off-diagonal terms: lambda
    Ce[0, 1] = Ce[0, 2] = Ce[1, 0] = Ce[1, 2] = Ce[2, 0] = Ce[2, 1] = lambda_modulus
    
    # Shear terms: mu
    Ce[3, 3] = Ce[4, 4] = Ce[5, 5] = mu_modulus
    
    return Ce

# Randomly generate Ludwik-Voce parameters
def generate_ludwik_voce_params():
    # Generate Ludwik-Voce parameters within a typical range
    a = np.random.uniform(0.5, 1.0)
    l1 = np.random.uniform(100, 1000)
    l2 = np.random.uniform(100, 500)
    l3 = np.random.uniform(0.1, 1.0)
    v1 = np.random.uniform(100, 500)
    v2 = np.random.uniform(500, 1000)
    v3 = np.random.uniform(5, 20)
    return a, l1, l2, l3, v1, v2, v3

# Compute yield stress using the Ludwik-Voce hardening model
def sigma_yield(p, a, l1, l2, l3, v1, v2, v3, sigma_y0):
    # Yield stress depends on accumulated plastic strain p
    return sigma_y0 + a * (l1 + l2 * p**l3) + (1 - a) * (v1 + (v2 - v1) * np.exp(-p / v3))

# Compute the hardening modulus (derivative of yield stress w.r.t p)
def hardening_modulus(p, a, l1, l2, l3, v1, v2, v3):
    # Derivative of sigma_yield with respect to p
    return a * (l2 * l3 * p**(l3 - 1)) + (1 - a) * (-v2 + v1) * (1 / v3) * np.exp(-p / v3)

# def the hardening jacobian
def hardening_jacobian(delta_p, sigma_e_trial, G, a, l1, l2, l3, v1, v2, v3, sigma_y0, p_n):
    """
    Jacobian function for root-finding of delta_p.
    """
    # Compute updated p
    p = p_n + delta_p

    # Compute hardening modulus
    h = hardening_modulus(p, a, l1, l2, l3, v1, v2, v3)

    # Compute the Jacobian
    return -3 * G - h

# Compute the deviatoric stress direction
def deviatoric_stress_direction(sigma_D):
    # Normalize deviatoric stress tensor
    return np.sqrt(3 / 2) * sigma_D / np.linalg.norm(sigma_D)

# Newton-Raphson method to compute p_{n+1}
def newton_raphson_update(sigma_trial, p_n, a, l1, l2, l3, v1, v2, v3, sigma_y0, max_iter=50, tol=1e-6):
    # Initialize p_k with the previous plastic strain p_n
    p_k = p_n  
    for _ in range(max_iter):
        # Compute yield stress and residual function f
        sigma_y = sigma_yield(p_k, a, l1, l2, l3, v1, v2, v3, sigma_y0)
        f = sigma_trial - sigma_y
        # Check for convergence
        if abs(f) < tol:
            break
        # Compute hardening modulus h and update p_k
        h = hardening_modulus(p_k, a, l1, l2, l3, v1, v2, v3)
        p_k -= f / (-h)  # Newton-Raphson update
    return p_k

# Randomly generate elastic constants
def generate_elastic_constants():
    # Randomly generate Young's modulus (E) and Poisson's ratio (nu)
    E = np.random.uniform(50e3, 300e3)  # Young's modulus in MPa
    nu = np.random.uniform(0.25, 0.35)  # Poisson's ratio
    # Convert to Lame constants
    lambda_modulus = E * nu / ((1 + nu) * (1 - 2 * nu))  # First Lame constant
    mu_modulus = E / (2 * (1 + nu))  # Shear modulus
    return lambda_modulus, mu_modulus

# Generate training data
def generate_training_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        # Generate Ludwik-Voce parameters
        a, l1, l2, l3, v1, v2, v3 = generate_ludwik_voce_params()
        
        # Generate elastic constants
        lambda_modulus, mu_modulus = generate_elastic_constants()
        
        # Initial conditions
        sigma_y0 = np.random.uniform(50, 400)  # Initial yield stress
        sigma_n = np.random.uniform(0, 500)  # Current stress
        p_n = np.random.uniform(0, 0.2)  # Accumulated plastic strain
        delta_epsilon = np.random.uniform(1e-5, 1e-2)  # Strain increment
        
        # Compute trial stress (assuming elastic behavior initially)
        sigma_trial = sigma_n + 2 * mu_modulus * delta_epsilon  # Trial stress
        
        # Update accumulated plastic strain using Newton-Raphson
        p_n1 = newton_raphson_update(sigma_trial, p_n, a, l1, l2, l3, v1, v2, v3, sigma_y0)
        
        # Compute plasticity-related increments
        delta_lambda = p_n1 - p_n  # Plastic multiplier
        delta_epsilon_p = delta_lambda  # Plastic strain increment (simplified)
        delta_epsilon_e = delta_epsilon - delta_epsilon_p  # Elastic strain increment
        delta_sigma = 2 * mu_modulus * delta_epsilon_e  # Stress increment
        
        # Update stress
        sigma_n1 = sigma_n + delta_sigma
        
        # Compute tangent modulus (approximation)
        h = hardening_modulus(p_n1, a, l1, l2, l3, v1, v2, v3)
        ddsdde = 2 * mu_modulus * h / (h + 2 * mu_modulus)  # Tangent modulus
        
        # Save all the data
        data.append([sigma_n, delta_epsilon, p_n, sigma_y0, p_n1, delta_epsilon_p,
                     delta_epsilon_e, delta_sigma, sigma_n1, ddsdde, lambda_modulus, mu_modulus])
    
    return np.array(data)

# def yield_ressidual(p, sigma_trial, a, l1, l2, l3, v1, v2, v3, sigma_y0):
#     """Residual function: consistency condition f(p) = sigma_trial - sigma_y(p)"""
#     sigma_y = sigma_yield(p, a, l1, l2, l3, v1, v2, v3, sigma_y0)
#     return sigma_trial - sigma_y

def yield_residual(delta_p, sigma_e_trial, mu, a, l1, l2, l3, v1, v2, v3, sigma_y0, p_n):
    """
    Residual function for yield condition based on the given formula:
    f = sigma_e_trial - 3G * delta_p - sigma_y
    """
    # Compute the updated plastic strain: p = p_n + delta_p
    p = p_n + delta_p

    # Compute yield stress based on Ludwik-Voce hardening
    sigma_y = sigma_yield(p, a, l1, l2, l3, v1, v2, v3, sigma_y0)
    # Residual equation
    return sigma_e_trial - 3 * mu * delta_p - sigma_y


def nonlinear_solver(residual_func, initial_guess, jacobian=None, solver_options=None, method='hybr', *args, **kwargs):
    """
    General nonlinear solver using scipy.optimize.root.
    
    Parameters:
    -----------
    residual_func : callable
        The residual function `f(x)` to be solved, where `f(x) = 0`.
        Should take `x` as the first argument, followed by `*args`.
    initial_guess : array_like
        Initial guess for the solution `x`.
    jacobian : callable, optional
        Jacobian function `J(x)` for `f(x)` (if available). Should return
        `J(x)` as a square matrix.
    solver_options : dict, optional
        Dictionary of options to pass to the solver. Examples:
        - `xtol`: Solution tolerance (default depends on method).
        - `maxiter`: Maximum number of iterations (for certain methods).
    method : str, optional
        Solver method to use. Options include:
        - 'hybr' (default): Hybrid Powell method, suitable for most problems.
        - 'lm': Levenberg-Marquardt method (good for least-squares).
        - 'broyden1': Broyden’s first method (good for large sparse problems).
    *args, **kwargs:
        Additional arguments to pass to `residual_func`.

    Returns:
    --------
    solution : array_like
        The solution `x` such that `residual_func(x) ≈ 0`.
    info : OptimizeResult
        A dictionary with details of the optimization process, including:
        - `success`: Whether the solver converged.
        - `message`: Reason for termination.
        - `nfev`: Number of function evaluations.
        - Other method-specific details.
    
    Raises:
    -------
    ValueError
        If the solver does not converge.
    """
    # Call the root solver
    result = root(residual_func, x0=initial_guess, jac=jacobian, method=method, options=solver_options, *args, **kwargs)

    # Check for convergence
    if result.success:
        return result.x, result
    else:
        raise ValueError(f"Solver did not converge: {result.message}")

import numpy as np

def computeDsDe(elasticity_tensor, n, h):
    """
    Compute the tangent stiffness matrix (jac_ep) for a 3D material 
    undergoing elastoplastic deformation.

    Parameters:
    -----------
    elasticity_tensor : ndarray
        Elastic stiffness matrix (4th-order tensor in Voigt notation).
        Typically a symmetric 6x6 matrix for isotropic or anisotropic materials.
    n : ndarray
        Flow direction vector, derived from the normalized deviatoric stress tensor.
        It has the same dimensionality as the stress space (e.g., 6 for 3D in Voigt notation).
    h : float
        Hardening modulus, representing the rate of isotropic hardening.
        A higher value indicates more resistance to plastic deformation.

    Returns:
    --------
    jac_ep : ndarray
        Tangent stiffness matrix (plastic stiffness matrix).
        This matrix incorporates both elastic stiffness and plastic flow effects
        to capture material behavior in the elastoplastic regime.
    """

    # The denominator includes the hardening modulus `h` and the projection
    # of the elastic stiffness along the flow direction.
    denominator = h + np.dot(np.dot(n.T, elasticity_tensor), n)

    # jac_ep adjusts the elastic stiffness by removing plastic flow contributions.
    # Outer product ensures the reduction is directional along `n`.
    jac_ep= elasticity_tensor - np.outer(np.dot(elasticity_tensor, n), np.dot(n, elasticity_tensor)) / denominator
    
    # Return the computed plastic stiffness matrix
    return jac_ep


# Generate training data

if __name__ == "__main__":
    # # Generate Ludwik-Voce parameters
    # a, l1, l2, l3, v1, v2, v3 = generate_ludwik_voce_params()
    
    # # Generate elastic constants
    # lambda_modulus, mu_modulus = generate_elastic_constants()
    
    # # Initial conditions
    # sigma_y0 = np.random.uniform(50, 400)  # Initial yield stress
    # sigma_n = np.random.uniform(0, 500)  # Current stress
    # p_n = np.random.uniform(0, 0.2)  # Accumulated plastic strain
    # delta_epsilon = np.random.uniform(1e-5, 1e-2)  # Strain increment
    
    # # Compute trial stress (assuming elastic behavior initially)
    # sigma_trial = sigma_n + 2 * mu_modulus * delta_epsilon  # Trial stress
    pass