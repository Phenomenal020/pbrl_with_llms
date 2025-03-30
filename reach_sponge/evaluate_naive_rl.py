    # Optionally, load and evaluate any agent
    # model = PPO.load(f"{models_dir}/reach-sponge_ppo_{5000}")
    # obs = env.reset()
    # episodes = 100
    # for _ in range(episodes):
    #     action, _states = model.predict(obs)
    #     obs, rewards, done, info = env.step(action)
    #     print(rewards)
    #     if done:
    #         obs = env.reset()