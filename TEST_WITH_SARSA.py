import numpy as np
import matplotlib.pyplot as plt

# Định nghĩa lớp Mê Cung
class Maze:
    def __init__(self, maze, start_position, goal_position):
        self.maze = maze
        self.maze_height = maze.shape[0]  # Chiều cao của mê cung (số hàng)
        self.maze_width = maze.shape[1]   # Chiều rộng của mê cung (số cột)
        self.start_position = start_position  # Vị trí bắt đầu
        self.goal_position = goal_position    # Vị trí mục tiêu

    def show_maze(self):
        plt.figure(figsize=(5, 5))
        plt.imshow(self.maze, cmap='gray')
        plt.text(self.start_position[1], self.start_position[0], 'S', ha='center', va='center', color='red', fontsize=20)
        plt.text(self.goal_position[1], self.goal_position[0], 'G', ha='center', va='center', color='green', fontsize=20)
        plt.xticks([]), plt.yticks([])
        plt.show()

# Định nghĩa các hành động có thể có
actions = [(-1, 0),  # Up
           (1, 0),   # Down
           (0, -1),  # Left
           (0, 1)]   # Right

# Định nghĩa lớp Tác Nhân SARSA
class SARSA_Agent:
    def __init__(self, maze, learning_rate=0.1, discount_factor=0.9, exploration_start=1.0, exploration_end=0.01, num_episodes=100):
        self.q_table = np.zeros((maze.maze_height, maze.maze_width, 4))  # 4 hành động
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_start = exploration_start
        self.exploration_end = exploration_end
        self.num_episodes = num_episodes

    def get_exploration_rate(self, current_episode):
        exploration_rate = self.exploration_start * (self.exploration_end / self.exploration_start) ** (current_episode / self.num_episodes)
        return exploration_rate

    def get_action(self, state, current_episode):
        exploration_rate = self.get_exploration_rate(current_episode)
        if np.random.rand() < exploration_rate:
            return np.random.randint(4)  # Hành động ngẫu nhiên
        else:
            return np.argmax(self.q_table[state])  # Hành động tốt nhất từ bảng Q

    def update_q_table(self, state, action, next_state, next_action, reward):
        current_q_value = self.q_table[state][action]
        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * self.q_table[next_state][next_action] - current_q_value)
        self.q_table[state][action] = new_q_value

# Các tham số phần thưởng
wall_penalty = -10
goal_reward = 10
step_penalty = -1

# Kết thúc một tập huấn luyện
def finish_episode_sarsa(agent, maze, current_episode, train=True):
    current_state = maze.start_position
    action = agent.get_action(current_state, current_episode)  # Chọn hành động đầu tiên
    is_done = False
    episode_reward = 0
    path = [current_state]

    while not is_done:
        next_state = (current_state[0] + actions[action][0], current_state[1] + actions[action][1])

        # Kiểm tra tường hoặc ra ngoài
        if (next_state[0] < 0 or next_state[0] >= maze.maze_height or
            next_state[1] < 0 or next_state[1] >= maze.maze_width or
            maze.maze[next_state] == 1):
            reward = wall_penalty
            next_state = current_state  # Giữ nguyên trạng thái
        elif next_state == maze.goal_position:
            path.append(next_state)
            reward = goal_reward
            is_done = True
        else:
            path.append(next_state)
            reward = step_penalty

        episode_reward += reward

        if train:
            next_action = agent.get_action(next_state, current_episode)  # Chọn hành động tiếp theo
            agent.update_q_table(current_state, action, next_state, next_action, reward)

        current_state = next_state
        action = next_action if train else action  # Giữ hành động cho lần lặp tiếp theo

    return episode_reward, path

# Huấn luyện tác nhân SARSA
def train_agent_sarsa(agent, maze, num_episodes=100):
    episode_rewards = []
    for episode in range(num_episodes):
        episode_reward, path = finish_episode_sarsa(agent, maze, episode, train=True)
        episode_rewards.append(episode_reward)

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.xlabel('Tập')
    plt.ylabel('Phần thưởng tích lũy')
    plt.title('Phần thưởng mỗi tập')
    average_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"Phần thưởng trung bình là: {average_reward}")
    plt.show()

# Kiểm tra tác nhân sau khi huấn luyện
def test_agent(agent, maze):
    current_state = maze.start_position
    is_done = False
    path = [current_state]

    while not is_done:
        action = agent.get_action(current_state, current_episode=0)  # Không cần khám phá trong giai đoạn kiểm tra
        next_state = (current_state[0] + actions[action][0], current_state[1] + actions[action][1])

        # Kiểm tra tường hoặc ra ngoài
        if (next_state[0] < 0 or next_state[0] >= maze.maze_height or
            next_state[1] < 0 or next_state[1] >= maze.maze_width or
            maze.maze[next_state] == 1):
            next_state = current_state  # Giữ nguyên trạng thái

        path.append(next_state)
        if next_state == maze.goal_position:
            is_done = True

        current_state = next_state

    return path

# Vẽ đường đi của tác nhân
def plot_path(maze, path):
    plt.figure(figsize=(5, 5))
    plt.imshow(maze.maze, cmap='gray')
    plt.text(maze.start_position[1], maze.start_position[0], 'S', ha='center', va='center', color='red', fontsize=20)
    plt.text(maze.goal_position[1], maze.goal_position[0], 'G', ha='center', va='center', color='green', fontsize=20)
    for position in path:
        plt.text(position[1], position[0], "#", va='center', color='blue', fontsize=20)
    plt.xticks([]), plt.yticks([])
    plt.grid(color='black', linewidth=2)
    plt.title('Đường đi của tác nhân')
    plt.show()
def test_agent_optimized(agent, maze):
    current_state = maze.start_position
    is_done = False
    path = [current_state]

    while not is_done:
        action = np.argmax(agent.q_table[current_state])  # Chọn hành động tốt nhất
        next_state = (current_state[0] + actions[action][0], current_state[1] + actions[action][1])

        # Kiểm tra tường hoặc ra ngoài
        if (next_state[0] < 0 or next_state[0] >= maze.maze_height or
            next_state[1] < 0 or next_state[1] >= maze.maze_width or
            maze.maze[next_state] == 1):
            next_state = current_state  # Giữ nguyên trạng thái

        path.append(next_state)
        if next_state == maze.goal_position:
            is_done = True

        current_state = next_state

    return path

# Huấn luyện agent SARSA
maze_layout = np.array([[0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 1, 1, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 1, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 0, 1, 0]])
maze = Maze(maze_layout,(0,0),(5,5))
maze.show_maze()

sarsa_agent = SARSA_Agent(maze)
train_agent_sarsa(sarsa_agent, maze, num_episodes=100)

# Kiểm tra tác nhân với đường đi tối ưu
optimized_path = test_agent_optimized(sarsa_agent, maze)
print("Đường đi tối ưu của tác nhân từ vị trí bắt đầu đến vị trí mục tiêu:", optimized_path)

# Vẽ đường đi tối ưu
plot_path(maze, optimized_path)
