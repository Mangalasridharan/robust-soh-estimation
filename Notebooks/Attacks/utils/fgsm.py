import tensorflow as tf

@tf.function
def fgsm_attack_batch(model, X_ic, X_ctx, y, epsilon, clip_min, clip_max, scaler_std):
    """
    FGSM attack for batched inputs - works directly with (batch, 20, 1) shape
    """
    # Ensure consistent dtype
    X_ic = tf.cast(X_ic, tf.float32)
    X_ctx = tf.cast(X_ctx, tf.float32)
    y = tf.cast(y, tf.float32)
    scaler_std = tf.cast(scaler_std, tf.float32)
    epsilon = tf.cast(epsilon, tf.float32)
    clip_min = tf.cast(clip_min, tf.float32)
    clip_max = tf.cast(clip_max, tf.float32)

    # Ensure scaler_std has compatible shape for broadcasting
    if len(scaler_std.shape) == 1:
        scaler_std = tf.reshape(scaler_std, (1, 20, 1))  # (20,) → (1, 20, 1)
    
    # Forward pass + gradient computation
    with tf.GradientTape() as tape:
        tape.watch(X_ic)
        y_pred = model([X_ic, X_ctx], training=False)
        mse_loss = tf.reduce_mean(tf.square(y - y_pred))
    
    # Compute gradient w.r.t. X_ic
    grad = tape.gradient(mse_loss, X_ic)  # Shape: (batch, 20, 1)
    
    # Scale epsilon to normalized space
    epsilon_scaled = epsilon / scaler_std  # Broadcasts to (batch, 20, 1)
    
    # FGSM perturbation - NOTE THE NEGATIVE SIGN! ✅
    perturbation = -epsilon_scaled * tf.sign(grad)  # ← ADDED MINUS SIGN
    
    # Apply perturbation
    X_adv = X_ic + perturbation
    
    # Clip to valid range
    X_adv = tf.clip_by_value(X_adv, clip_min, clip_max)
    
    return X_adv
