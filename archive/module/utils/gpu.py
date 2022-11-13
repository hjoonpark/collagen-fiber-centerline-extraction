import GPUtil

class GPUStat:
    def get_stat(self):
        gpus = [gpu for gpu in GPUtil.getGPUs()]
        stats = []
        for gpu in gpus:
            s = {}
            s['name'] = gpu.name
            s['id'] = gpu.id
            s['mem_total'] = gpu.memoryTotal
            s['mem_free'] = gpu.memoryFree # MB
            s['mem_used'] = gpu.memoryUsed
            s['mem_util'] = gpu.memoryUtil * 100 # %
            s['temperature'] = gpu.temperature # C
            stats.append(s)
        return stats

    def get_stat_str(self):
        gpus = [gpu for gpu in GPUtil.getGPUs()]
        stat_strs = []
        for gpu in gpus:
            s = "  [{}] {}, Memory: total({:,.1f} GB) used({:,.1f} GB) free({:,.1f} GB) {:,.0f}% | temperature({:.0f} 'C)".format(gpu.id, gpu.name, gpu.memoryTotal/1000.0, gpu.memoryUsed/1000.0, gpu.memoryFree/1000.0, gpu.memoryUtil*100, gpu.temperature)
            stat_strs.append(s)
        return '\n'.join(stat_strs)
