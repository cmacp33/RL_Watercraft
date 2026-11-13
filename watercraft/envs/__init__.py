"""Small factory around the unchanged Gazebo environment and physical boat."""

def make(settings, operation, seed):
    environment = settings[operation]['environment']
    if environment == 'sim':
        from .sim import make as make_sim
        return make_sim(settings, seed)
    from .real import RealEnv
    return RealEnv(settings, operation, seed)


def stop(env):
    if getattr(env.unwrapped, 'is_real', False):
        env.stop()
    else:
        from .sim import stop as stop_sim
        stop_sim(env)


def close(env):
    if getattr(env.unwrapped, 'is_real', False):
        env.close()
    else:
        from .sim import close as close_sim
        close_sim(env)
