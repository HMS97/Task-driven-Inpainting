import matplotlib.pyplot as plt
import os 

def plot_generated(image_dic, save_dir = './', name = 'example'):
    """
    send input your image list for plotting and return figure
    save_dir: path to save output
    name: name for the save image name 
    """
    # assert len(image_list) == 3, "please check your output images!"
    fig=plt.figure(figsize=(20, 20))

    for index, item in enumerate(image_dic.items()):
        fig.add_subplot(1, len(image_dic), index+1)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(item[0], fontsize=18 )
        plt.imshow(item[1])

    plt.savefig(os.path.join(save_dir, f"{name}.png") ,bbox_inches='tight')
    plt.close()
    return fig


def plot_seg(image_list):
    """
    send input your image list for plotting and return figure
    """
    assert len(image_list) == 2, "please check your output images!"
    fig=plt.figure(figsize=(20, 20))
    fig.add_subplot(1, 2, 1)
    plt.xlabel('target', fontsize=18 )
    plt.imshow(image_list[0])
    fig.add_subplot(1, 2, 2)
    plt.xlabel('predict', fontsize=18 )
    plt.imshow(image_list[1])
    plt.savefig("example_seg.png",bbox_inches='tight')

    return fig