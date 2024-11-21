from train_data.gen_traindata import *
import numpy as np

def test_assemble_elasticity_tensor():
    """Test the correctness of the 3D elastic stiffness tensor."""
    lambda_modulus = 100
    mu_modulus = 50

    # Expected result based on manual computation
    expected = np.array([
        [200, 100, 100,   0,   0,   0],
        [100, 200, 100,   0,   0,   0],
        [100, 100, 200,   0,   0,   0],
        [  0,   0,   0,  50,   0,   0],
        [  0,   0,   0,   0,  50,   0],
        [  0,   0,   0,   0,   0,  50],
    ])

    # Call the function
    result = assemble_elasticity_tensor(lambda_modulus, mu_modulus)

    # Assert the result matches the expected values
    assert result.shape == (6, 6), f"Expected shape (6, 6), but got {result.shape}"
    assert np.allclose(result, expected), "Computed elasticity tensor does not match expected values"

def test_nonlinear_solver_ludwik_voce():
    # Example parameters
    G = 80e3  # Shear modulus (MPa)
    sigma_e_trial = 600  # Trial equivalent stress (MPa)
    p_n = 0.05  # Previous accumulated plastic strain

    # Ludwik-Voce parameters
    a = 0.8
    l1 = 200
    l2 = 100
    l3 = 0.2
    v1 = 300
    v2 = 600
    v3 = 20
    sigma_y0 = 200  # Initial yield stress (MPa)

    # Initial guess for delta_p
    initial_guess = 0.0

    # Solve for delta_p using the generalized nonlinear solver
    delta_p, info = nonlinear_solver(
        residual_func=yield_residual,
        initial_guess=initial_guess,
        jacobian=hardening_jacobian,
        solver_options={'xtol': 1e-8},  # Tolerance for convergence
        args=(sigma_e_trial, G, a, l1, l2, l3, v1, v2, v3, sigma_y0, p_n)
    )

    # Compute the updated stress
    sigma_updated = sigma_e_trial - 3 * G * delta_p
    sigma_y = sigma_yield(p_n + delta_p, a, l1,l2,l3, v1, v2, v3, sigma_y0)
    print("delta p is: ", delta_p, 
          "updated flow stress is: ", sigma_updated,
          "yield stress is: ", sigma_y,
          "yield residual is: ", sigma_updated - sigma_y)
    assert np.isclose(sigma_updated - sigma_y, 0), "delta p solve failed"

def test_computeDsDe():
    """
    Test the Jacobian function of stress w.r.t. strain using a 6x6 elastic stiffness tensor 
    in Voigt notation.
    """
    # Define material parameters
    lambda_modulus = 100e3  # Lame's first parameter (MPa)
    mu_modulus = 50e3       # Shear modulus (MPa)
    
    # Assemble the 6x6 elastic stiffness tensor
    elasticity_tensor = assemble_elasticity_tensor(lambda_modulus, mu_modulus)

    # Define flow direction (6D in Voigt notation)
    n = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Simple uniaxial case

    # Define hardening modulus
    h = 10e3  # Hardening modulus (MPa)

    # Call the computeQpJacobian function
    jac_ep = computeDsDe(elasticity_tensor, n, h)

    # Test: Symmetry of the Jacobian
    is_symmetric = np.allclose(jac_ep, jac_ep.T)
    assert is_symmetric, "Jacobian matrix is not symmetric"

    # Test: Numerical stability (no NaN or Inf values)
    assert not np.any(np.isnan(jac_ep)), "Jacobian matrix contains NaN values"
    assert not np.any(np.isinf(jac_ep)), "Jacobian matrix contains Inf values"

    # Test: Large h (Jacobian should approximate the elasticity tensor)
    large_h = 1e12  # Very high hardening modulus
    jac_ep_large_h = computeDsDe(elasticity_tensor, n, large_h)
    assert np.allclose(jac_ep_large_h, elasticity_tensor), (
        "Jacobian does not match elasticity tensor for large h"
    )

    # Test: Small h (Perfectly plastic case)
    small_h = 1e-9  # Very low hardening modulus
    jac_ep_small_h = computeDsDe(elasticity_tensor, n, small_h)
    denominator_small_h = small_h + np.dot(np.dot(n.T, elasticity_tensor), n)
    expected_jac_ep_small_h = (
        elasticity_tensor
        - np.outer(np.dot(elasticity_tensor, n), np.dot(n, elasticity_tensor)) / denominator_small_h
    )
    assert np.allclose(jac_ep_small_h, expected_jac_ep_small_h), (
        "Jacobian mismatch for small h (perfectly plastic case)"
    )




    