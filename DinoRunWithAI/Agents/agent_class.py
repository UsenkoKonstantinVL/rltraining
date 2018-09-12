class Agent:
    def remember(self, state, action, reward, next_state, done): pass

    def act(self, state): pass

    def clear_act(self, state): pass

    def replay(self, batch_size): pass

    def load(self): pass

    def save(self): pass

    def update_target_model(self): pass
