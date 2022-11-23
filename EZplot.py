import matplotlib.pyplot as plt
import matplotlib.patches as pch

def plot_single(plot_list, legend, opts="r", size=(16,6), filename=None):
    plt.figure(figsize=size)
    
    plt.plot(plot_list, opts)
    
    info = pch.Patch(color=opts[0], label=legend) #color=opts[0] kicsit sufnis
    plt.legend(handles=[info], loc="upper left")
    
    if filename != None:
        plt.savefig(filename)
    plt.show()
    
def plot_multi(plots, lgd_list, opt_list, size=(16,6), filename=None):
    plt.figure(figsize=size)
    info_list = []
    
    for idx, plot_list in enumerate(plots):
        plt.plot(plot_list, opt_list[idx])
        
        info = pch.Patch(color=opt_list[idx][0], label=lgd_list[idx]) #color=opt_list[idx][0] kicsit sufnis
        info_list.append(info)
    plt.legend(handles=info_list, loc="upper left")
    
    if filename != None:
        plt.savefig(filename)
    plt.show()