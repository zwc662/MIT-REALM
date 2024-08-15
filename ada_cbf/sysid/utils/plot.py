import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
 
# Function to draw the plot
def draw_plot(waypoints, obs_pointss, all_sim_traj_points, filename):
    plt.figure()
    plt.scatter(waypoints['x'], waypoints['y'], color='black', label='WPs', s = .05)
    # Plot observation points in red
    plt.scatter(obs_pointss[0]['x'][0], obs_pointss[0]['y'][0], color='black', s = 120, label='Traj')
    for i, obs_points in enumerate(obs_pointss):
        for j in range(len(obs_points['x']) - 1):
            plt.arrow(obs_points['x'][j], obs_points['y'][j],
                        obs_points['x'][j+1] - obs_points['x'][j+1], obs_points['y'][j] - obs_points['y'][j],
                        head_width=0.05, head_length=0.1, fc='black', ec='black')
        plt.scatter(obs_points['x'][-1], obs_points['y'][-1], color='black', s = 60 if i < len(obs_points) - 1 else 120)
    

    # Create a color map
    colors = cm.rainbow(np.linspace(0, 1, len(all_sim_traj_points)))
    
    # Plot each simulated trajectory with a different color
    for i, sim_traj_points in enumerate(all_sim_traj_points):
        plt.scatter(sim_traj_points['x'][0], sim_traj_points['y'][0], color=colors[i], s = 30, marker='^')
        for j in range(len(sim_traj_points['x']) - 1):
            plt.arrow(sim_traj_points['x'][j], sim_traj_points['y'][j],
                        sim_traj_points['x'][j+1] - sim_traj_points['x'][j], sim_traj_points['y'][j+1] - sim_traj_points['y'][j],
                        head_width=0.05, head_length=0.1, fc=colors[i], ec=colors[i])
        plt.scatter(sim_traj_points['x'][-1], sim_traj_points['y'][-1], color=colors[i], label=f'Sim Traj {i+1}', s = 30)
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('2D Plot of Observations and Simulated Trajectories')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make space on the right for the legend
    plt.savefig(filename)

 