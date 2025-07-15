def validate_model(model, input_dim=128, dead_threshold=0.01):
    """Comprehensive model health check with dead neuron detection"""
    device = next(model.parameters()).device
    
    # Test inputs
    zero_input = torch.zeros(4, input_dim, device=device).half()
    rand_input = torch.randn(4, input_dim, device=device).half()
    strong_input = torch.ones(4, input_dim, device=device).half() * 3
    
    # Forward passes
    model.eval()
    with torch.no_grad():
        # Test different input strengths
        _ = model(zero_input)
        zero_activity = model.a_bar.clone()
        
        _ = model(rand_input)
        rand_activity = model.a_bar.clone()
        
        _ = model(strong_input)
        strong_activity = model.a_bar.clone()
    
    # 1. Check output ranges
    states = model.x
    assert not torch.isnan(states).any(), "NaNs in states!"
    assert states.min() > -10 and states.max() < 10, "States exploding!"
    
    # 2. Activity response check
    zero_mean = zero_activity.mean().item()
    rand_mean = rand_activity.mean().item()
    strong_mean = strong_activity.mean().item()
    print(f"Activity response: Zero={zero_mean:.4f}, Random={rand_mean:.4f}, Strong={strong_mean:.4f}")
    assert rand_mean > zero_mean + 0.05, "No input response!"
    assert strong_mean > rand_mean + 0.1, "No intensity response!"
    
    # 3. Feature distance sensitivity
    dists = torch.cdist(model.features.data, model.features.data)
    distance_sensitivity = True
    for i in range(5):  # Check first 5 neurons
        similar_idx = torch.argmin(dists[i])
        dissimilar_idx = torch.argmax(dists[i])
        if model.W[i, similar_idx].item() <= model.W[i, dissimilar_idx].item():
            print(f"⚠️ Warning: Neuron {i} has weak connection to similar neuron!")
            distance_sensitivity = False
    assert distance_sensitivity, "Distance sensitivity broken!"
    
    # 4. Dynamic behavior
    state_changes = []
    test_input = torch.randn(4, input_dim, device=device).half()
    for t in range(20):
        _ = model(test_input)
        state_changes.append(model.x.std().item())
    
    print("State change std:", np.std(state_changes))
    assert 0.01 < np.std(state_changes) < 0.5, "Abnormal dynamics!"
    
    # 5. Dead neuron analysis (CRITICAL)
    print("\n=== Dead Neuron Analysis ===")
    
    # Calculate neuron activity statistics
    activities = torch.stack([zero_activity, rand_activity, strong_activity])
    max_activity = activities.amax(dim=(0,1))  # Max per neuron across all tests
    mean_activity = activities.mean(dim=(0,1))  # Mean per neuron
    
    # Identify dead neurons
    dead_mask = (max_activity < dead_threshold)
    dead_count = dead_mask.sum().item()
    dead_percentage = 100 * dead_count / model.N
    
    print(f"Dead neurons: {dead_count}/{model.N} ({dead_percentage:.1f}%)")
    print(f"Dead neuron indices: {torch.where(dead_mask)[0].cpu().numpy()}")
    
    # Check mean activity of dead neurons
    if dead_count > 0:
        dead_means = mean_activity[dead_mask]
        print(f"Mean activity of dead neurons: {dead_means.cpu().numpy()}")
    
    # Visualize activity distribution
    plt.figure(figsize=(10, 6))
    plt.hist(max_activity.cpu().numpy(), bins=50, alpha=0.7)
    plt.axvline(dead_threshold, color='r', linestyle='--', label='Dead Threshold')
    plt.title("Maximum Neuron Activity Distribution")
    plt.xlabel("Activity Level")
    plt.ylabel("Neuron Count")
    plt.legend()
    plt.show()
    
    # Dead neuron thresholds
    assert dead_percentage < 20.0, f"Too many dead neurons: {dead_percentage:.1f}% > 20%"
    assert dead_count < 50, f"Excessive dead neurons: {dead_count} > 50"
    
    print("✅ All health checks passed!")
    return True

# Run validation
validate_model(model, dead_threshold=0.05)  # Adjust threshold as needed
