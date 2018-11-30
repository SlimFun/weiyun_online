
COMPUTE_ABILITY = 1.0

def T(path, entry_task, exit_task, w_vex):
    t = 0
    if exit_task < 0 or exit_task > len(path):
        return 0
    for i in range(entry_task, exit_task):
        t += w_vex[path[i]] / COMPUTE_ABILITY
    return t