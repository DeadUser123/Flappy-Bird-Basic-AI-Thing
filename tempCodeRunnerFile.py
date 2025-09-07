class FlappyBirdEnv: # simulate flappy bird (was too lazy to find one to pull off from internet)
    def __init__(self, episode=0):
        self.gravity = 0.5
        self.flap_strength = -8.0
        self.bird_x = 50.0
        self.bird_y = 250.0
        self.bird_vel = 0.0
        self.pipe_width = 50
        self.screen_width = 400
        self.screen_height = 500
        self.pipes = []
        self.score = 0
        self.bird_radius = 12

        # curriculum to make things easier early on (because the thing kept dying a lot)
        self.episode = episode
        self.easy_gap = 200
        self.normal_gap = 120
        self.pipe_gap = self.easy_gap if episode < 1000 else self.normal_gap

        # state: normalized features (bird_y, bird_vel, x_dist_to_pipe, y_diff_to_gap_center, pipe_y)
        self.state_size = 5
        self.action_size = 2

        self.reset()

    def reset(self):
        self.bird_y = 250.0
        self.bird_vel = 0.0
        gap_margin = int(self.pipe_gap // 2 + 20)
        self.pipes = [[200.0, float(random.randint(gap_margin, self.screen_height - gap_margin))]]
        self.score = 0
        self._passed_current = False
        return self.get_state()

    def get_state(self):
        pipe_x, pipe_y = self.pipes[0]
        dx = (pipe_x - self.bird_x) / float(self.screen_width) # distance between bird and pipe
        dy = (self.bird_y - pipe_y) / float(self.screen_height) # vertical distance between bird and pipe center
        vy = np.clip(self.bird_vel / 10.0, -1.0, 1.0) # bird vertical velocity
        by = np.clip(self.bird_y / float(self.screen_height), 0.0, 1.0) # bird y position
        py = np.clip(pipe_y / float(self.screen_height), 0.0, 1.0)  # pipe y position (forgor that dy existed)
        return np.array([by, vy, np.clip(dx, -1.0, 1.0), np.clip(dy, -1.0, 1.0), py], dtype=np.float32)

    def step(self, action): # basically next frame
        if action == 1:
            self.bird_vel = self.flap_strength
        self.bird_vel += self.gravity
        self.bird_y += self.bird_vel

        for p in self.pipes:
            p[0] -= 3.0

        reward = 0.01
        if action == 1:
            reward -= 0.01

        pipe_x, pipe_y = self.pipes[0]
        gap_top = pipe_y - self.pipe_gap / 2.0
        gap_bot = pipe_y + self.pipe_gap / 2.0
        done = False

        bird_left = self.bird_x - self.bird_radius
        bird_right = self.bird_x + self.bird_radius

        # Collision
        if (pipe_x < bird_right and pipe_x + self.pipe_width > bird_left):
            if not (gap_top < self.bird_y < gap_bot):
                done = True
                reward = -10.0

        if self.bird_y <= 0 or self.bird_y >= self.screen_height:
            done = True
            reward = -10.0

        # Enhanced shaping: reward based on distance to gap center to encourage being alive
        gap_center = pipe_y
        dist_to_center = abs(self.bird_y - gap_center)
        reward += max(0.0, (self.pipe_gap / 2 - dist_to_center) / (self.pipe_gap / 2)) * 0.1

        # Scoring
        if pipe_x + self.pipe_width < self.bird_x and not self._passed_current:
            self.score += 1
            reward += 1.0
            self._passed_current = True

        # Recycle pipe
        if pipe_x < -self.pipe_width:
            self.pipes.pop(0)
            gap_margin = int(self.pipe_gap // 2 + 20)
            self.pipes.append([float(self.screen_width), float(random.randint(gap_margin, self.screen_height - gap_margin))])
            self._passed_current = False
            self.pipe_gap = self.normal_gap

        reward = float(np.clip(reward, -20.0, 20.0))
        return self.get_state(), reward, done, {}