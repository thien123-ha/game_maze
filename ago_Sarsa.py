import numpy as np
import pygame
import time
import matplotlib.pyplot as plt

# Lớp Maze để hiển thị và làm việc với bố cục mê cung
class Maze:
    def __init__(self, maze, start_position, goal_position):
        self.maze = maze
        self.maze_height = maze.shape[0]
        self.maze_width = maze.shape[1]
        self.start_position = start_position
        self.goal_position = goal_position

    def show_maze(self, path=None):
        pygame.init()
        cell_size = 60
        window_size = (self.maze_width * cell_size, self.maze_height * cell_size)
        screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption('Maze Visualization')

        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)

        def draw_maze():
            for row in range(self.maze_height):
                for col in range(self.maze_width):
                    color = WHITE if self.maze[row][col] == 0 else BLACK
                    pygame.draw.rect(screen, color, (col * cell_size, row * cell_size, cell_size, cell_size))

            pygame.draw.rect(screen, RED, (self.start_position[1] * cell_size, self.start_position[0] * cell_size, cell_size, cell_size))
            pygame.draw.rect(screen, GREEN, (self.goal_position[1] * cell_size, self.goal_position[0] * cell_size, cell_size, cell_size))

        def draw_path(path):
            if path:
                for pos in path:
                    draw_maze()
                    pygame.draw.rect(screen, BLUE, (pos[1] * cell_size, pos[0] * cell_size, cell_size, cell_size))
                    pygame.draw.rect(screen, RED, (self.start_position[1] * cell_size, self.start_position[0] * cell_size, cell_size, cell_size))
                    pygame.draw.rect(screen, GREEN, (self.goal_position[1] * cell_size, self.goal_position[0] * cell_size, cell_size, cell_size))
                    pygame.display.update()
                    time.sleep(1)

        running = True
        while running:
            screen.fill(WHITE)
            draw_maze()
            if path:
                draw_path(path)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        pygame.quit()

# Khởi tạo mê cung
maze_layout = np.array([[0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 1, 1, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 1, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 0, 1, 0]])
maze = Maze(maze_layout, (0, 0), (5, 5))

actions = [(-1, 0),  # Lên
           (1, 0),   # Xuống
           (0, -1),  # Trái
           (0, 1)]   # Phải

class SARSAAgent:
    def __init__(self, maze, learning_rate=0.1, discount_factor=0.9, exploration_start=1.0, exploration_end=0.01, num_episodes=1000):
        self.q_table = np.zeros((maze.maze_height, maze.maze_width, 4))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_start = exploration_start
        self.exploration_end = exploration_end
        self.num_episodes = num_episodes

    def get_exploration_rate(self, current_episode):
        return self.exploration_start * (self.exploration_end / self.exploration_start) ** (current_episode / self.num_episodes)

    def get_action(self, state, current_episode):
        exploration_rate = self.get_exploration_rate(current_episode)
        if np.random.rand() < exploration_rate:
            return np.random.randint(4)  # Chọn hành động ngẫu nhiên
        else:
            return np.argmax(self.q_table[state])  # Chọn hành động có giá trị Q lớn nhất

    def update_q_table(self, state, action, next_state, next_action, reward):
        current_q_value = self.q_table[state][action]
        next_q_value = self.q_table[next_state][next_action]
        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * next_q_value - current_q_value)
        self.q_table[state][action] = new_q_value

goal_reward = 100
wall_penalty = -10
step_penalty = -1

def finish_episode(agent, maze, current_episode, train=True):
    current_state = maze.start_position
    current_action = agent.get_action(current_state, current_episode)
    is_done = False
    episode_reward = 0
    episode_step = 0
    collision_count = 0
    path = [current_state]

    while not is_done:
        next_state = (current_state[0] + actions[current_action][0], current_state[1] + actions[current_action][1])

        if (next_state[0] < 0 or next_state[0] >= maze.maze_height or 
            next_state[1] < 0 or next_state[1] >= maze.maze_width or 
            maze.maze[next_state[0]][next_state[1]] == 1):
            reward = wall_penalty
            next_state = current_state
            collision_count += 1
        elif next_state == maze.goal_position:
            path.append(next_state)
            reward = goal_reward
            is_done = True
        else:
            path.append(next_state)
            reward = step_penalty

        if train:
            next_action = agent.get_action(next_state, current_episode) if not is_done else None
            agent.update_q_table(current_state, current_action, next_state, next_action if next_action is not None else 0, reward)

        current_state = next_state
        current_action = next_action if not is_done else None
        episode_reward += reward
        episode_step += 1

    print("episode_step:", episode_step, "collisions:", collision_count, "episode_reward:", episode_reward)
    
    return episode_reward, episode_step, path

def train_agent(agent, maze, num_episodes=100):
    rewards = []
    for episode in range(num_episodes):
        episode_reward, _, _ = finish_episode(agent, maze, episode, train=True)
        rewards.append(episode_reward)
    return rewards

def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Rewards over Episodes during Training')
    plt.show()

def test_agent(agent, maze):
    current_state = maze.start_position
    path = [current_state]
    is_done = False

    while not is_done:
        action = np.argmax(agent.q_table[current_state])
        next_state = (current_state[0] + actions[action][0], current_state[1] + actions[action][1])

        if next_state == maze.goal_position:
            path.append(next_state)
            is_done = True
        else:
            path.append(next_state)
        
        current_state = next_state

    maze.show_maze(path)

# Khởi tạo tác nhân SARSA
agent = SARSAAgent(maze)

# Huấn luyện tác nhân
rewards = train_agent(agent, maze, num_episodes=1000)

# Kiểm tra tác nhân sau khi huấn luyện
test_agent(agent, maze)

# Hiển thị biểu đồ phần thưởng
plot_rewards(rewards)

# In danh sách phần thưởng sau huấn luyện
print("list_rewards", rewards)

