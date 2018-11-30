from weiyun.utils.log_utils import logger

class Cloud:
    def __init__(self, core_num, compute_ability=4.0):
        self.core_num = core_num
        self.compute_ability = compute_ability

    def T(self, path, entry_task, exit_task, w_vex):
        t = 0
        for i in range(entry_task, exit_task+1):
            t += w_vex[path[i]] / self.compute_ability
        return t

    def reorder_modules(self, modules, topo_order):
        # print(modules)
        logger.debug('modules : %s'.format(modules))
        r = []
        for t in topo_order:
            if t in modules:
                r.append(t)
        return r

    def check_anc_scheduled(self, ancs, i, un_scheduled):
        for v in ancs[i]:
            if v in un_scheduled:
                return False
        return True

    # queue : [[9, 3, 0, 1], [7, 4, 2, -1]]
    def schedule_queue(self, v_tag, topo_order, anc_matrix):
        queue = []
        modules = []
        for i in range(len(v_tag)):
            if v_tag[i] == 1:
                modules.append(i)
        for i in range(self.core_num):
            queue.append([])
        ms = self.reorder_modules(modules, topo_order)
        ancs = {}
        for i in ms:
            ancs.update({i:[]})
            for j in ms:
                if anc_matrix[j][i] == 1:
                    ancs[i].append(j)
        # print(ancs)
        logger.debug('ancs : %s'.format(ancs))
        un_scheduled = ms

        list = []
        core = []
        for i in range(self.core_num):
            core.append(i)
        i = 0
        while len(un_scheduled) != 0:
            if i % self.core_num == 0:
                for v in list:
                    un_scheduled.remove(v)
                list = []
            for un_v in un_scheduled:
                if len(list) == self.core_num:
                    break
                    # completion_queue(queue)
                    # sq = shortest_queue(queue)
                    # for s in sq:
                    #     queue[s].append(0)
                if self.check_anc_scheduled(ancs, un_v, un_scheduled):
                    list.append(un_v)
                    queue[core[i % self.core_num]].append(un_v)
                    i += 1
            if i % self.core_num != 0 and len(un_scheduled) != 0:
                self.completion_queue(queue)
                while i % self.core_num != 0:
                    i += 1
        # for i in range(len(ms)):
        #     for un_v in un_scheduled:
        #         if
        # for i in range(len(ms)):
        #     queue[i % CORE_NUM].append(ms[i])
        # print('queue : {0}'.format(queue))
        logger.debug('queue : {0}'.format(queue))
        return queue

    def completion_queue(self, queue):
        len_qs = []
        for q in queue:
            len_qs.append(len(q))
        sq = []
        s = min(len_qs)
        for i in range(len(len_qs)):
            if len_qs[i] == s:
                sq.append(i)
        if len(sq) != len(len_qs):
            for q in sq:
                queue[q].append(-1)