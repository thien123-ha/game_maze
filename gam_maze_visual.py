import numpy as np
import pygame
import time

import matplotlib.pyplot as plt

# Lớp Maze để hiển thị và làm việc với bố cục mê cung
class Maze:
    def __init__(self, maze, start_position, goal_position):
        self.maze = maze
        # Lấy chiều cao và chiều rộng của mê cung
        self.maze_height = maze.shape[0]  # Số hàng trong mê cung
        self.maze_width = maze.shape[1]   # Số cột trong mê cung
        # Đặt vị trí bắt đầu và vị trí mục tiêu dưới dạng tuple (tọa độ x, y)
        self.start_position = start_position
        self.goal_position = goal_position

    # Phương thức để hiển thị mê cung sử dụng pygame
    def show_maze(self, path=None):
        pygame.init()
        cell_size = 60 # Kích thước ô vuông trong mê cung
        window_size = (self.maze_width * cell_size, self.maze_height * cell_size)  # Kích thước cửa sổ
        screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption('Maze Visualization')

        # Định nghĩa màu sắc
        BLACK = (0, 0, 0)       # Màu đen cho tường
        WHITE = (255, 255, 255) # Màu trắng cho đường đi
        RED = (255, 0, 0)       # Màu đỏ cho vị trí bắt đầu
        GREEN = (0, 255, 0)     # Màu xanh lá cho vị trí mục tiêu
        BLUE = (0, 0, 255)      # Màu xanh dương cho đường đi của tác nhân

        # Hàm để vẽ mê cung lên màn hình
        def draw_maze():
            for row in range(self.maze_height):
                for col in range(self.maze_width):
                    color = WHITE if self.maze[row][col] == 0 else BLACK
                    pygame.draw.rect(screen, color, (col * cell_size, row * cell_size, cell_size, cell_size))

            # Đánh dấu vị trí bắt đầu và mục tiêu
            pygame.draw.rect(screen, RED, (self.start_position[1] * cell_size, self.start_position[0] * cell_size, cell_size, cell_size))
            pygame.draw.rect(screen, GREEN, (self.goal_position[1] * cell_size, self.goal_position[0] * cell_size, cell_size, cell_size))

        # Hàm để vẽ đường đi của tác nhân
        '''def draw_path(path):
            if path:
                for pos in path:
                    pygame.draw.rect(screen, BLUE, (pos[1] * cell_size, pos[0] * cell_size, cell_size, cell_size))
                    pygame.display.update()
                    time.sleep(0.01)  # Dừng 0.3 giây để thấy tác nhân di chuyển'''
        def draw_path(path):
            if path:
                for i, pos in enumerate(path):
            # Vẽ lại mê cung để xóa các vị trí trước đó
                    draw_maze()
            
            # Vẽ vị trí hiện tại của tác nhân với màu xanh
                    pygame.draw.rect(screen, BLUE, (pos[1] * cell_size, pos[0] * cell_size, cell_size, cell_size))
            
            # Đánh dấu lại vị trí bắt đầu và mục tiêu (để không bị mất khi vẽ lại mê cung)
                    pygame.draw.rect(screen, RED, (self.start_position[1] * cell_size, self.start_position[0] * cell_size, cell_size, cell_size))
                    pygame.draw.rect(screen, GREEN, (self.goal_position[1] * cell_size, self.goal_position[0] * cell_size, cell_size, cell_size))

                    pygame.display.update()  # Cập nhật màn hình sau khi vẽ
                    time.sleep(1)  # Dừng lại 0.01 giây để thấy tác nhân di chuyển
        # Vòng lặp chính của pygame để vẽ mê cung
        running = True
        episode_count = 0  # Đếm số tập

        while running :
                screen.fill(WHITE)
                draw_maze()
                if path:
                    draw_path(path)
                pygame.display.update()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                episode_count += 1  # Tăng số tập sau mỗi lần lặp

        pygame.quit()


# Khởi tạo mê cung
'''maze_layout = np.array([[0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 0, 1, 0],
                        [1, 1, 0, 1, 0],
                        [0, 0, 0, 0, 0]])

# Tạo một đối tượng mê cung
maze = Maze(maze_layout, (0, 0), (4, 4))'''
'''maze_layout = np.array([[0,0,0,0,0,0,1,1,0,0],[0,0,0,1,0,0,0,0,1,0],
                        [0,0,0,0,1,1,0,0,1,1],[0,0,1,0,0,1,0,1,1,0],
                        [0,1,0,0,0,0,0,1,0,0],[0,1,0,0,0,0,0,0,0,1],
                        [0,0,0,1,0,1,1,1,0,0],[0,0,0,1,0,0,0,0,1,0],
                        [1,0,0,1,0,1,0,1,1,0],[1,1,0,0,0,0,0,0,0,0]]) # tao duong di va vat can(0 dai dien cho duong di),(1 dai dien cho vat can )
maze = Maze(maze_layout,(0,0),(9,9))'''
maze_layout = np.array([[0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 1, 1, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 1, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 0, 1, 0]])
maze = Maze(maze_layout,(0,0),(5,5))
actions = [(-1, 0),  # Lên
           (1, 0),   # Xuống
           (0, -1),  # Trái
           (0, 1)]   # Phải
# Lớp tác nhân Q-learning
class QLearningAgent:
    def __init__(self, maze, learning_rate=0.1, discount_factor=0.9, exploration_start=1.0, exploration_end=0.01, num_episodes=100):
        self.q_table = np.zeros((maze.maze_height, maze.maze_width, 4))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_start = exploration_start
        self.exploration_end = exploration_end
        self.num_episodes = num_episodes

    # Hàm xác định mức khám phá (exploration) từ 1.0 đến 0.01
    def get_exploration_rate(self, current_episode):
        return self.exploration_start * (self.exploration_end / self.exploration_start) ** (current_episode / self.num_episodes)

    # Hàm chọn hành động
    def get_action(self, state, current_episode):
        exploration_rate = self.get_exploration_rate(current_episode)
        if np.random.rand() < exploration_rate:
            return np.random.randint(4)  # Chọn hành động ngẫu nhiên
        else:
            return np.argmax(self.q_table[state])  # Chọn hành động có giá trị Q lớn nhất
    
    # Cập nhật bảng Q-Table
    def update_q_table(self, state, action, next_state, reward):
        best_next_action = np.argmax(self.q_table[next_state])
        current_q_value = self.q_table[state][action]
        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * self.q_table[next_state][best_next_action] - current_q_value)
        self.q_table[state][action] = new_q_value
        

# Hệ thống phần thưởng
goal_reward = 100
wall_penalty = -10
step_penalty = -1

# Hàm hoàn thành một tập (episode)
def finish_episode(agent, maze, current_episode, train=True):
    current_state = maze.start_position
    is_done = False
    episode_reward = 0
    episode_step = 0
    collision_count = 0  # Đếm số lần va chạm với tường
    path = [current_state]

    while not is_done:
        action = agent.get_action(current_state, current_episode)
        next_state = (current_state[0] + actions[action][0], current_state[1] + actions[action][1])

        # Kiểm tra va chạm với tường hoặc vượt ngoài mê cung
        if (next_state[0] < 0 or next_state[0] >= maze.maze_height or 
            next_state[1] < 0 or next_state[1] >= maze.maze_width or 
            maze.maze[next_state[0]][next_state[1]] == 1):
            reward = wall_penalty
            next_state = current_state
            collision_count += 1  # Tăng số lần va chạm

        # Đạt đến mục tiêu
        elif next_state == maze.goal_position:
            path.append(next_state)
            reward = goal_reward
            is_done = True
        else:
            path.append(next_state)
            reward = step_penalty

        episode_reward += reward
        episode_step += 1

        if train:
            agent.update_q_table(current_state, action, next_state, reward)

        current_state = next_state

    print("episode_step:", episode_step, "collisions:", collision_count ,"episode_reward:", episode_reward)
    
    


    return episode_reward, episode_step, path


# Định nghĩa các hành động (lên, xuống, trái, phải)


# Hàm huấn luyện tác nhân Q-learning
def train_agent(agent, maze, num_episodes=100):
    rewards = [] 
    for episode in range(num_episodes):
        episode_reward, _, _ = finish_episode(agent, maze, episode, train=True)
        rewards.append(episode_reward)  # Lưu phần thưởng của mỗi tập vào danh sách
    return rewards

# Hàm hiển thị biểu đồ phần thưởng theo từng tập
def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Rewards over Episodes during Training')
    plt.show()

# Kiểm tra tác nhân sau khi huấn luyện
def test_agent(agent, maze):
    episode_reward, episode_step, path = finish_episode(agent, maze, current_episode=0, train=False)
    maze.show_maze(path)

# Khởi tạo tác nhân và mê cung
agent = QLearningAgent(maze)

# Huấn luyện tác nhân
rewards = train_agent(agent, maze, num_episodes=100)


# Kiểm tra tác nhân sau khi huấn luyện
test_agent(agent, maze)

# Hiển thị biểu đồ phần thưởng
plot_rewards(rewards)

# In danh sách phần thưởng sau huấn luyện
print("list_rewards", rewards)

 

