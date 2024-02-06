import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# csv_file = r'C:\Users\bensc\PycharmProjects\scikit\to_image\tracking\final_position_results.csv'

csv_file = r'C:\Users\bensc\PycharmProjects\scikit\to_image\tracking\active_id_trajectory.csv'
df = pd.read_csv(csv_file)
# df = df.drop_duplicates()

print(df)
fig, ax = plt.subplots()

def update_plot(i):
    ax.clear()  # Clear the previous plot
    ax.scatter(df['x_pos'][:i+1], df['y_pos'][:i+1])  # Plot x and y up to frame i
    ax.set_title(f'Time: {df["most_recent_frame"][i]}')  # Set title with corresponding time
    ax.invert_yaxis()
    ax.axhline(y=70, color='blue', linestyle='-')
    ax.axhline(y=460, color='red', linestyle='--')
    ax.axhline(y=780, color='blue', linestyle='-')

# for obj_id in range(len(df)):
#     filtered_df = df.loc[(df['object_id'] == obj_id)]
#     plt.plot(filtered_df['most_recent_frame'], filtered_df['y_pos'], 'go-', label=str(obj_id), linewidth=2)



# title = 'Substrate Fluorescence Response\nat Varied MMP-9 Concentrations'
xaxis = 'most_recent_frame'
yaxis = 'y_pos'

#Font
# font = {'size': 20, 'fontweight':'bold'} #BOLD Version
font = {'size': 18}

#Change title
# plt.title(title, fontdict = font)

#Change axis labels
plt.xlabel(xaxis, fontdict = font)
plt.ylabel(yaxis, fontdict = font)

anim = FuncAnimation(fig, update_plot, frames=len(df), interval=100)

plt.show()


# TODO: Figure out how to track trajectories from entire video. Not just finishing positions