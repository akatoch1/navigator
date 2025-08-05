import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TopDownNavEnvContinuous(gym.Env):

    def __init__(
        self,
        arena_size: float = 5.0,
        max_step_size: float = 0.20,  
        max_turn_rate: float = math.pi / 12, 
        goal_radius: float = 0.25,
        max_steps: int = 500,
        n_obs: int = 1,
        obs_radius: float = 0.35,
        n_rays: int = 16,
        lidar_range: float = 6.0
    ):
        super().__init__()
        self.arena_size = arena_size
        self.max_step_size = max_step_size
        self.max_turn_rate = max_turn_rate
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.n_obs = n_obs
        self.obs_radius = obs_radius
        self.n_rays = n_rays
        self.lidar_range = lidar_range

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        high = np.ones(n_rays + 4, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

       
        if seed is not None:
            self.seed(seed)

    def seed(self, seed: int):
        np.random.RandomState().seed(seed)

    def sample(self):
        if self.n_obs == 0:
            return np.array([], dtype=np.float32).reshape(0, 2)
        
        to_goal = self._goal - self._agent_pos
        distance_to_goal = np.linalg.norm(to_goal)
        
        if distance_to_goal < 2.0:
            return np.array([], dtype=np.float32).reshape(0, 2)
        
        t = np.random.RandomState().uniform(0.3, 0.7)
        obstacle_pos = self._agent_pos + t * to_goal
        
        perpendicular = np.array([-to_goal[1], to_goal[0]])
     
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        offset = np.random.RandomState().uniform(-0.5, 0.5) * self.obs_radius
        obstacle_pos += offset * perpendicular
        
        obstacle_pos = np.clip(obstacle_pos, -self.arena_size * 0.9, self.arena_size * 0.9)
        
        return np.array([obstacle_pos], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)

        self._agent_pos = np.random.RandomState().uniform(-1.0, 1.0, 2) * self.arena_size * 0.25
        self._theta = np.random.RandomState().uniform(-math.pi, math.pi)
        while True:
            self._goal = np.random.RandomState().uniform(-self.arena_size * 0.9, self.arena_size * 0.9, 2)
            if np.linalg.norm(self._goal - self._agent_pos) > 2.5:
                break

        self._obstacles = self.sample()
        self._step_counter = 0
        self._prev_dist = np.linalg.norm(self._goal - self._agent_pos)
        return self.get_objs(), {}

    def detect(self, angle_world: float) -> float:
        p0 = self._agent_pos
        v = np.array([math.cos(angle_world), math.sin(angle_world)])
        max_t = self.lidar_range
        min_hit = max_t
        for c in self._obstacles:
            o = p0 - c
            b = 2 * np.dot(v, o)
            cval = np.dot(o, o) - self.obs_radius ** 2
            disc = b ** 2 - 4 * cval
            if disc < 0:
                continue
            sqrt_disc = math.sqrt(disc)
            t1 = (-b - sqrt_disc) / 2.0
            t2 = (-b + sqrt_disc) / 2.0
            for t in (t1, t2):
                if 0 < t < min_hit:
                    min_hit = t
        return min_hit / max_t

    def get_objs(self):
        rays = []
        start_ang = -math.pi / 2  
        for i in range(self.n_rays):
            a = start_ang + i * math.pi / (self.n_rays - 1)
            rays.append(self.detect(self._theta + a))
        to_goal = self._goal - self._agent_pos
        c, s = math.cos(-self._theta), math.sin(-self._theta)
        gx = (c * to_goal[0] - s * to_goal[1]) / self.arena_size
        gy = (s * to_goal[0] + c * to_goal[1]) / self.arena_size
        gx, gy = np.clip([gx, gy], -1.0, 1.0)
        obs = np.concatenate([rays, [gx, gy, math.sin(self._theta), math.cos(self._theta)]]).astype(np.float32)
        return obs

    
    def collides(self, pos):
        if not (-self.arena_size <= pos[0] <= self.arena_size and -self.arena_size <= pos[1] <= self.arena_size):
            return True
        
        if any(np.linalg.norm(pos - c) <= self.obs_radius for c in self._obstacles):
            return True
        return False

 
    def step(self, action: np.ndarray):
        self._step_counter += 1
        collision = False

   
        action = np.clip(action, -1.0, 1.0)
        forward_speed, turn_rate = action

        self._theta += turn_rate * self.max_turn_rate

        if abs(forward_speed) > 1e-6:  
            step_size = forward_speed * self.max_step_size
            direction = np.array([math.cos(self._theta), math.sin(self._theta)])
            new_pos = self._agent_pos + step_size * direction
            
            if self.collides(new_pos):
                collision = True
               
            else:
                self._agent_pos = new_pos

  
        dist = np.linalg.norm(self._goal - self._agent_pos)
        reached_goal = dist < self.goal_radius
        progress = self._prev_dist - dist
        
        reward = progress * 5.0
        
        reward -= 0.005
        
        action_penalty = 0.001 * (forward_speed**2 + turn_rate**2)
        reward -= action_penalty
        
        if collision:
            reward -= 25.0
            
        if reached_goal:
            reward += 50.0
            
        self._prev_dist = dist

        terminated = reached_goal or collision
        truncated = self._step_counter >= self.max_steps

        info = {"is_success": reached_goal,
                "termination_reason": "goal" if reached_goal else "collision" if collision else "timeout" if truncated else "running"}
        return self.get_objs(), reward, terminated, truncated, info

    


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


env = TopDownNavEnvContinuous(n_obs=1)  


model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    n_steps=1024, 
    batch_size=256,
    learning_rate=3e-4,
    clip_range=0.2,
    ent_coef=0.01,  
    vf_coef=0.5,   
)

model.learn(total_timesteps=200_000)
model.save("ppo_topdown_nav_continuous")
    
  
   
