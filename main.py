
from CAM_heatmap import draw_CAM
from Draw_heatmap import draw_features



if __name__ == '__main__':


    ### Node that 
    # heat is the tensor to be visited
    # form : [1, C, H ,W ]
    # s is the prediction result  after sigmoid or softmax 
    # form : [0.5,0.1,0.2,...]


    # draw_features(16,32,heat.cpu().numpy())


    # draw_CAM(heat, s, root+name, './')### need grad
    pass