import tensorflow as tf
import numpy as np

@tf.function
def fgsm_attack_batch(model, x_ic, x_ctx, y, scaler_std, epsilon_physical, clip_min, clip_max):
    """
    FGSM attack for batched inputs
    
    Args:
        model: Trained keras model
        x_ic: IC curve input (scaled) - shape: (batch_size, 20)
        x_ctx: Context features - shape: (batch_size, n_features)
        y: True SOH values - shape: (batch_size, 1)
        scaler_std: Standard deviation from training scaler - shape: (20,)
        epsilon_physical: Perturbation magnitude in physical units (Ah/V) - scalar
        clip_min: Minimum allowed value for IC curves (scaled space)
        clip_max: Maximum allowed value for IC curves (scaled space)
    
    Returns:
        Adversarial IC curves - shape: (batch_size, 20)
    """
    # Cast inputs
    x_ic = tf.cast(x_ic, tf.float32)
    x_ctx = tf.cast(x_ctx, tf.float32)
    y = tf.cast(y, tf.float32)
    
    # Track gradient
    with tf.GradientTape() as tape:
        tape.watch(x_ic)
        y_pred = model([x_ic, x_ctx], training=False)
        # MSE loss over batch
        loss = tf.reduce_mean(tf.square(y - y_pred))
    
    # Get gradient (shape: batch_size, 20)
    grad = tape.gradient(loss, x_ic)
    
    # Safety check
    if grad is None:
        tf.print("WARNING: Gradient is None. Check model connections.")
        return x_ic
    
    # Convert scaler_std to tensor and reshape for broadcasting against x_ic rank.
    # Example:
    # x_ic (B, F)    -> scaler (1, F)
    # x_ic (B, F, 1) -> scaler (1, F, 1)
    scaler_std_tensor = tf.cast(tf.convert_to_tensor(scaler_std), tf.float32)
    x_rank = tf.rank(x_ic)
    tail_ones = tf.ones((x_rank - 2,), dtype=tf.int32)
    scaler_shape = tf.concat(([1, tf.shape(scaler_std_tensor)[0]], tail_ones), axis=0)
    scaler_std_tensor = tf.reshape(scaler_std_tensor, scaler_shape)
    
    # Calculate scaled epsilon for each feature
    # epsilon_physical / scaler_std_tensor gives shape: (1, 20)
    epsilon_physical = tf.cast(epsilon_physical, tf.float32)
    epsilon_scaled = epsilon_physical / scaler_std_tensor
    
    # Generate perturbation and adversarial example
    # tf.sign(grad) shape: (batch_size, 20)
    # epsilon_scaled shape: (1, 20) broadcasts to (batch_size, 20)
    perturbation = epsilon_scaled * tf.sign(grad)
    x_adv = x_ic + perturbation
    
    # Clip to valid range
    clip_min = tf.cast(clip_min, tf.float32)
    clip_max = tf.cast(clip_max, tf.float32)
    x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)
    
    return x_adv


def pgd_attack(model, x_ic_batch, x_ctx_batch, y_batch,
                   epsilon, alpha, num_iter,
                   clip_min, clip_max):
    """
    GPU-optimized PGD attack (iterative FGSM with projection)
    
    Args:
        epsilon: Maximum perturbation (L∞ ball radius)
        alpha: Step size per iteration
        num_iter: Number of iterations
    """
    # Ensure tensors
    x_ic_batch = tf.convert_to_tensor(x_ic_batch, dtype=tf.float32)
    x_ctx_batch = tf.convert_to_tensor(x_ctx_batch, dtype=tf.float32)
    y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
    
    # Initialize adversarial examples
    x_ic_adv = tf.Variable(x_ic_batch, dtype=tf.float32)
    x_ic_orig = x_ic_batch
    
    for i in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(x_ic_adv)
            predictions = model([x_ic_adv, x_ctx_batch], training=False)
            loss = tf.reduce_mean(tf.square(y_batch - predictions))
        
        gradient = tape.gradient(loss, x_ic_adv)
        
        # Update: x = x + alpha * sign(gradient)
        x_ic_adv.assign_add(alpha * tf.sign(gradient))
        
        # Project back to epsilon ball around original
        perturbation = x_ic_adv - x_ic_orig
        perturbation = tf.clip_by_value(perturbation, -epsilon, epsilon)
        x_ic_adv.assign(x_ic_orig + perturbation)
        
        # Clip to valid range
        x_ic_adv.assign(tf.clip_by_value(x_ic_adv, clip_min, clip_max))
    
    return x_ic_adv.read_value()

import tensorflow as tf


@tf.function  # enables graph mode + GPU acceleration
def jsma_regression_attack_gpu(
    model,
    x_ic_batch,
    x_ctx_batch,
    y_batch,
    max_iterations=10,     # number of JSMA feature steps
    theta=0.1,             # max total perturbation per feature (L0 budget per feature)
    gamma=0.01,            # step size per iteration
    clip_min=-5.0,
    clip_max=2.0
):
    """
    Vectorized JSMA-style untargeted regression attack (GPU-optimized).

    Args:
        model : tf.keras model taking [x_ic, x_ctx]
        x_ic_batch : (B, F) attackable features
        x_ctx_batch : (B, C) context features (not perturbed)
        y_batch : (B, 1) ground truth regression targets
    """

    # Convert to tensors (GPU-friendly)
    x_ic = tf.convert_to_tensor(x_ic_batch, tf.float32)
    x_ctx = tf.convert_to_tensor(x_ctx_batch, tf.float32)
    y_true = tf.convert_to_tensor(y_batch, tf.float32)

    # Adversarial variable
    x_adv = tf.Variable(x_ic)

    # Track |perturbation| per feature → ensures L0-style bounded updates
    feature_budget = tf.Variable(tf.zeros_like(x_ic))

    for _ in tf.range(max_iterations):

        with tf.GradientTape() as tape:
            tape.watch(x_adv)

            y_pred = model([x_adv, x_ctx], training=False)

            # Untargeted regression loss (maximize prediction error)
            error = tf.abs(y_pred - y_true)
            loss = tf.reduce_mean(tf.square(error))

        grads = tape.gradient(loss, x_adv)

        # ---------- Stable saliency (regression-aware) ----------
        # JSMA-style importance: |∂error/∂x| scaled by current error
        saliency = tf.abs(grads) * tf.stop_gradient(error)

        # ---------- Enforce per-feature perturbation budget ----------
        available = feature_budget < theta
        saliency = tf.where(available, saliency, tf.zeros_like(saliency))

        # ---------- L0 sparsity: choose ONE feature per sample ----------
        best_feature = tf.argmax(saliency, axis=1, output_type=tf.int32)

        # Gather gradient sign for chosen feature
        batch_indices = tf.range(tf.shape(x_adv)[0], dtype=tf.int32)
        gather_idx = tf.stack([batch_indices, best_feature], axis=1)

        direction = tf.sign(tf.gather_nd(grads, gather_idx))

        # ---------- Build sparse perturbation tensor ----------
        perturb = tf.scatter_nd(
            gather_idx,
            gamma * direction,
            tf.shape(x_adv)
        )

        # Apply perturbation
        x_adv.assign_add(perturb)

        # Update per-feature |delta|
        feature_budget.assign_add(tf.abs(perturb))

        # Clip to valid feature range
        x_adv.assign(tf.clip_by_value(x_adv, clip_min, clip_max))

    return x_adv


@tf.function
def cw_attack(
    model,
    x_ic,
    x_ctx,
    y,
    c=10.0,
    lr=0.01,
    steps=200,
    clip_min=-3.0,
    clip_max=3.0,
):
    """
    Untargeted Carlini & Wagner (C&W) attack for regression models.
    GPU-friendly TensorFlow implementation.

    Objective:
        minimize ||δ||_2^2  -  c · MSE(y_pred, y_true)

    where:
        δ = adversarial perturbation
    """

    # Ensure float32 tensors (important for GPU)
    x_ic = tf.cast(x_ic, tf.float32)
    x_ctx = tf.cast(x_ctx, tf.float32)
    y = tf.cast(y, tf.float32)

    # -------------------------------------------------
    # Step 1: Normalize input to [0, 1] for tanh trick
    # -------------------------------------------------
    x_norm = (x_ic - clip_min) / (clip_max - clip_min)

    # Avoid numerical issues at boundaries
    x_norm = tf.clip_by_value(x_norm, 1e-6, 1.0 - 1e-6)

    # -------------------------------------------------
    # Step 2: Initialize optimization variable using atanh
    # (starts exactly at original sample → minimal perturbation)
    # -------------------------------------------------
    w = tf.Variable(tf.atanh(2.0 * x_norm - 1.0))

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # -------------------------------------------------
    # Step 3: Optimization loop
    # -------------------------------------------------
    for _ in tf.range(steps):

        with tf.GradientTape() as tape:

            # Tanh transformation → keeps adversarial sample in [0,1]
            x_adv_norm = 0.5 * (tf.tanh(w) + 1.0)

            # Map back to original scaled feature range
            x_adv = x_adv_norm * (clip_max - clip_min) + clip_min

            # Model prediction
            y_pred = model([x_adv, x_ctx], training=False)

            # -------------------------------------------------
            # Proper squared L2 perturbation norm (original CW)
            # -------------------------------------------------
            perturb = x_adv_norm - x_norm
            l2 = tf.reduce_mean(
                tf.reduce_sum(tf.square(perturb), axis=[1, 2])
            )

            # -------------------------------------------------
            # Regression attack objective (untargeted)
            # maximize prediction error
            # -------------------------------------------------
            mse = tf.reduce_mean(tf.square(y - y_pred))

            # Final CW loss
            loss = l2 - c * mse

        grads = tape.gradient(loss, w)
        optimizer.apply_gradients([(grads, w)])

    # -------------------------------------------------
    # Step 4: Construct final adversarial example
    # -------------------------------------------------
    x_adv_norm = 0.5 * (tf.tanh(w) + 1.0)
    x_adv = x_adv_norm * (clip_max - clip_min) + clip_min

    # Ensure valid bounds
    x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)

    return x_adv
