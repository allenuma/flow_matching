from flow_imports import *
from flow_helpers import *


if __name__ == "__main__":


    device = 'cpu'
    num_epochs = 200
    sample_size = 512
    train_size = 512
    batch_size = 64
    dim_x = 2
    j = 24

    print(torch.cuda.is_available())
    # print(torch.cuda.current_device())



    x0 = sample_x0(sample_size=train_size, dim=dim_x)
    x1 = sample_x1(sample_size=train_size)
    y1 = measurement_operator(samples_x=x1)
    XY = construct_joint_dist(samples_x=x1, samples_y=y1)
    dataset_tensor = construct_training_data(x0, XY, j)

    
    train_dataset = TensorDataset(dataset_tensor)
    # train_batch_size = args.batch_size #num_train_samples if num_train_samples < 100 else 100
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = FlowMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()  # example loss

    step=0
    # pbar = tqdm(range(num_epochs), desc=f"Training", position=0, leave=True, colour='green')
    for epoch in tqdm(range(num_epochs), desc=f"Training", position=0, leave=True, colour='green'):
        # pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, X in enumerate(train_loader):
            model.train()
            step += 1

            x = X[0].to(device)

            v_pred, v_target = model(x)
            loss = loss_fn(v_pred, v_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # pbar.update(1)
            # pbar.set_postfix(loss=loss.item())
    



    # Initialize/Visualize the VF at the initial timestep
    t0 = 0.
    steps = 30  # 30x30 grid
    theta = 2*np.pi*torch.rand(1)
    measurement = torch.tensor([np.cos(theta),np.sin(theta)])+0.01*torch.randn(2)
    # measurement = torch.tensor([0,0.])
    device = 'cpu'
    input_tensor, grid_X, grid_Y = generate_grid_with_time(t=0., measurement=measurement, steps=steps, device='cpu')

    # New standard Gaussian samples and time column

    x0_k = sample_x0(sample_size=sample_size, dim=dim_x).reshape(sample_size, 2, 1)
    y_repeat = measurement.repeat(sample_size,1)
    XY = construct_joint_dist(samples_x=x0_k[:,:,int(t0)], samples_y=y_repeat)
    XY = XY.reshape((sample_size,4,1))
    t_values = repeated_linspace(0, 1, steps=j, repeats=sample_size).T

    
    model.eval()
    with torch.no_grad():
        v_k = model(input_tensor).reshape(steps*steps,2,1)  # (900, 2)

    for k in range(j):
        with torch.no_grad():
            
            # Generate input for model
            velocity = model(torch.cat([x0_k[:,:2,k], y_repeat, t_values.T[:,k].unsqueeze(-1)],dim=1))  # (900, 2)
            x0_kp1 = x0_k[:,:2,k] + (1/j)*velocity
            x0_k = torch.cat([x0_k, x0_kp1.reshape([sample_size,2,1])], dim=-1) 

            # Compute the grid Vector Field for visualization
            input_tensor, grid_X, grid_Y = generate_grid_with_time(t=k/j, measurement=measurement, steps=steps, device='cpu')
            v_k_next = model(input_tensor)  # (900, 2)
            v_k = torch.cat([v_k, v_k_next.reshape([steps*steps,2,1])], dim=-1)


            # U = v_k_next[:, 0].reshape(steps, steps)
            # V = v_k_next[:, 1].reshape(steps, steps)


    save_path = './flow_matching.mp4'
    animate_trajectories_and_vector_field(
        positions=x0_k,
        velocities=v_k,
        grid_X=grid_X,
        grid_Y=grid_Y,
        interval=100,
        x1=x1,
        y1=y1,
        measurement=measurement,
        save_path=save_path,  # or 'flow.mp4'
        frame_dir="output/frames"
    )