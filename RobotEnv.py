##Python 3.9.5 (tags/v3.9.5:0a7dcbd, May  3 2021, 17:27:52) [MSC v.1928 64 bit (AMD64)] on win32#
#Type "help", "copyright", "credits" or "license()" for more information.
"""import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class ObstacleRoomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.width = 10
        self.height = 10
        self.robot_radius = 1
        self.obstacle_radius = 1
        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]))
        self.observation_space = spaces.Box(np.array([0, 0, -np.pi]), np.array([self.width, self.height, np.pi]))
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.robot_pos = np.array([1, 1])
        self.goal_pos = np.array([self.width-2, self.height-2])
        self.obstacle_pos = np.array([[4, 4], [7, 7], [3, 6], [8, 3]])
        return self.get_observation()

    def get_observation(self):
        theta = np.arctan2(self.goal_pos[1] - self.robot_pos[1], self.goal_pos[0] - self.robot_pos[0])
        return np.concatenate((self.robot_pos, np.array([theta])))

    def step(self, action):
        self.robot_pos = np.clip(self.robot_pos + action, self.robot_radius, [self.width - self.robot_radius, self.height - self.robot_radius])
        done = np.linalg.norm(self.robot_pos - self.goal_pos) < self.robot_radius
        reward = -np.linalg.norm(self.robot_pos - self.goal_pos)
        for obstacle in self.obstacle_pos:
            if np.linalg.norm(self.robot_pos - obstacle) < self.robot_radius + self.obstacle_radius:
                done = True
                reward = -1
                break
        return self.get_observation(), reward, done, {}

    def render(self, mode='human'):
        print("Robot position: ({:.2f}, {:.2f})".format(self.robot_pos[0], self.robot_pos[1]))
        print("Goal position: ({:.2f}, {:.2f})".format(self.goal_pos[0], self.goal_pos[1]))
        print("Obstacle positions:")
        for obstacle in self.obstacle_pos:
            print("({:.2f}, {:.2f})".format(obstacle[0], obstacle[1]))

env = ObstacleRoomEnv()
observation = env.reset()
for t in range(100):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break"""

import gym
from gym import spaces
import numpy as np
import pygame

class RobotEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=255, shape=(100, 100, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        self.robot_position = np.array([50, 50])
        self.goal_position = np.array([80, 80])
        self.obstacle_positions = [np.array([30, 30]), np.array([70, 70])]
        
    def step(self, action):
        if action == 0:
            self.robot_position[0] -= 1
        elif action == 1:
            self.robot_position[0] += 1
        elif action == 2:
            self.robot_position[1] -= 1
        elif action == 3:
            self.robot_position[1] += 1
        
        reward = -1
        done = False
        if np.array_equal(self.robot_position, self.goal_position):
            reward = 100
            done = True
        elif np.any(np.all(self.robot_position == self.obstacle_positions, axis=1)):
            reward = -100
            done = True
        
        observation = np.zeros((100, 100, 3), dtype=np.uint8)
        observation[self.robot_position[0], self.robot_position[1]] = [255, 0, 0]
        observation[self.goal_position[0], self.goal_position[1]] = [0, 255, 0]
        for obstacle_position in self.obstacle_positions:
            observation[obstacle_position[0], obstacle_position[1]] = [0, 0, 255]
        
        return observation, reward, done, {}
    
    def reset(self):
        self.robot_position = np.array([50, 50])
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_x - self.min_x
        scale = screen_width/world_width
        robot_radius = self.robot_size/2.0

        # Initialize Pygame
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))

        # Draw the background
        self.screen.fill((255, 255, 255))

        # Draw the obstacles
        for obstacle in self.obstacles:
            obstacle_rect = pygame.Rect((obstacle.x - self.min_x) * scale,
                                        screen_height - (obstacle.y + obstacle.height) * scale,
                                        obstacle.width * scale,
                                        obstacle.height * scale)
            pygame.draw.rect(self.screen, (0, 0, 0), obstacle_rect)

        # Draw the robot
        robot_rect = pygame.Rect((self.robot.x - robot_radius - self.min_x) * scale,
                                screen_height - (self.robot.y + robot_radius) * scale,
                                self.robot_size * scale,
                                self.robot_size * scale)
        pygame.draw.rect(self.screen, (255, 0, 0), robot_rect)

        # Convert the Pygame surface to a numpy array
        img = pygame.surfarray.array3d(self.screen)

        # Normalize the image
        img = img / 255.0

        if mode == 'human':
            pygame.display.update()
        elif mode == 'rgb_array':
            return img
        else:
            super().render(mode=mode)


env = RobotEnv()
observation = env.reset()
for i in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()
    if done:
        observation = env.reset()
env.close()

