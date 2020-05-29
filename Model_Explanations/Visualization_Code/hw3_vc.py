import numpy as np
import tensorflow as tf
import hw3_utils


if __name__ == '__main__':
    data = hw3_utils.load_ImageNet()

    # print(data.X.shape)
    # print(data.Y.shape)
    # print(data.Pred.shape)
    # print(data.Idx.shape)

    # >>> Your code for written exercises here <<<
    '''Code for generating attribution maps is as follows'''
    image_set = load_ImageNet()
    '''Code used to figure out the VGG index of the images from the Image Net'''
    correct_image_idx = [12228, 12510, 12727, 1504, 11436]
    incorrect_image_idx = [2167, 10857, 1174, 11993, 10657]

    a = [np.where(image_set[3] == idx) for idx in correct_image_idx] #find where in Image set = the idx which corresponds to VGG idx
    b = [np.where(image_set[3] == idx) for idx in incorrect_image_idx]
    correct_idx_list_vgg_images = []
    incorrect_idx_list_vgg_images = []

    '''For some reason there were more than one image idx at certain locations using the function np.where so the code below
        was to get all the idx values into a list to then be viewed. '''
    correct_idx_list = [115, 1114, 517, 314, 908]
    print(correct_idx_list)
    correct_images = np.zeros((len(correct_idx_list), 224, 224, 3))
    correct_images = np.asarray([image_set[0][idx] for idx in correct_idx_list])
    print(correct_images.shape)
    for id in b:
        for j in range(len(id[0])):
            incorrect_idx_list_vgg_images.append(id[0][j])
    incorrect_idx_list = [717, 807, 107, 511, 4]
    # print(incorrect_idx_list)
    # for img in incorrect_idx_list:
    #     plt.imshow(image_set[0][img].reshape(224, 224, 3))
    #     plt.show()
    incorrect_images = np.zeros((len(incorrect_idx_list), 224, 224, 3))
    incorrect_images = np.asarray([image_set[0][idx] for idx in incorrect_idx_list])
    # print(incorrect_images.shape)

    '''Code for creation and generation of the Influence Directed Visualizations '''
    image_net_idx = [12228, 12510, 12727, 1504, 11436, 2167, 10857, 1174, 11993, 10657] #image Net idx
    idx_list = [115, 1114, 517, 314, 908, 717, 807, 107, 511, 4] #corresponding VGG idx
    imgs = np.array([image_set[0][idx] for idx in idx_list]) #get images from dataset corresponding to VGG idx
    pred_label = [image_set[2][idx].astype(int) for idx in idx_list] #pred labels to be used
    attr_map_combined = []
    for im_num in range(len(idx_list)):
        with tf.Session() as session:
            model = get_model(session)
            InflD = InflDirected(session, model.imgs, model.fc1, model.fc3l, lambda x: x[pred_label[im_num]]) #create Influence Directed object with specified target
            expert_unit_infl = InflD.dis_influence(image_set[0].astype(float), 1) #get expert unit
            print('Expert Neuron for all images: ', expert_unit_infl)
            attr_map_infl = InflD.expert_attribution(expert_unit_infl, imgs[im_num].reshape(1, 224, 224, 3).astype(float)) #get attribution map
            map_Infl = binary_mask(imgs[im_num].reshape(1, 224, 224, 3), attr_map_infl.reshape(1, 224, 224, 3)) #map to visualize
            pci = point_cloud(attr_map_infl.reshape(224, 224, 3)) #point cloud image
            subplot_list = [imgs[im_num].reshape(224, 224, 3), pci, map_Infl.reshape(224, 224, 3)]
            fig, axs = plt.subplots(1, 3, figsize=(10, 3))
            for ax, title, img in zip(axs, ['Original Image', 'Point Cloud', 'Binary Mask'], subplot_list):
                ax.imshow(img)
            if im_num < 5: #Correct Images
                fig.suptitle(["Influence Directed Map, Correct Classification, Image ID: ", image_net_idx[im_num]])
                savefile = "Influence_Directed_Map_Correct_" + str(image_net_idx[im_num]) + '.png'
            else: #Incorrect Images
                fig.suptitle(["Influence Directed Map, Incorrect Classification, Image ID: ", image_net_idx[im_num]])
                savefile = "Influence_Directed_Map_Incorrect_" + str(image_net_idx[im_num]) + '.png'
            plt.savefig(savefile)
            plt.show()


    '''The below is the code used to generate Integrated Grad Results. The code above is simply more refined.'''
    idx_list = [115, 1114, 517, 314, 908, 717, 807, 107, 511, 4] #VGG indices
    pred_label = [image_set[2][idx].astype(int) for idx in idx_list] #pred labels for target
    imgs = [image_set[0][idx].reshape(1, 224, 224, 3) for idx in idx_list] #imgs for idx_list
    IG_map_combined = []
    with tf.Session() as session:
        for i in range(len(idx_list)):
            model = get_model(session)
            IG3 = IntegratedGrad(session, model.imgs, model.fc3l, pred_label[i]) #Create IG object with target as the prediction of VGG model
            IG_map3 = IG3(imgs[i], 'black', 16, 50, True) #Call IG object to get the map
            IG_map_combined.append(IG_map3) #append to a list

    image_net_idx = [12228, 12510, 12727, 1504, 11436, 2167, 10857, 1174, 11993, 10657]
    point_cloud_IG3 = point_cloud(np.asarray(IG_map_combined)) #point cloud
    for idx, img in enumerate(imgs):
        map_IG = binary_mask(imgs[idx].reshape(1, 224, 224, 3), IG_map_combined[idx].reshape(1, 224, 224, 3)) #get IG map visual
        subplot_list = [imgs[idx].reshape(224, 224, 3), point_cloud_IG3[idx].reshape(224, 224, 3), map_IG.reshape(224, 224, 3)]
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        for ax, title, img in zip(axs, ['Original Image', 'Point Cloud', 'Binary Mask'], subplot_list):
            ax.imshow(img)
        if idx < 5:
            fig.suptitle(["Integrated Gradient Map, Correct Classification, Image ID: ", image_net_idx[idx]])
            savefile = "Integrated_Gradient_Map_Correct_" + str(image_net_idx[idx]) + '.png'
        else:
            fig.suptitle(["Integrated Gradient Map, Incorrect Classification, Image ID: ", image_net_idx[idx]])
            savefile = "Integrated_Gradient_Map_Incorrect_" + str(image_net_idx[idx]) + '.png'
        plt.savefig(savefile)
        plt.show()

    '''The code below is for Saliency Maps and is the same as the code used for Integrated Gradients but changed up savefile names'''
    idx_list = [115, 1114, 517, 314, 908, 717, 807, 107, 511, 4]
    pred_label = [image_set[2][idx].astype(int) for idx in idx_list]
    imgs = [image_set[0][idx].reshape(1, 224, 224, 3) for idx in idx_list]
    saliency_map_combined = []
    with tf.Session() as session:
        for i in range(len(idx_list)):
            model = get_model(session)
            salience3 = SaliencyMap(session, model.imgs, model.fc3l, pred_label[i])
            saliency_map3 = salience3(imgs[i], 16, True)
            saliency_map_combined.append(saliency_map3)

    image_net_idx = [12228, 12510, 12727, 1504, 11436, 2167, 10857, 1174, 11993, 10657]
    point_cloud_saliency = point_cloud(np.asarray(saliency_map_combined))
    for idx, img in enumerate(imgs):
        map_saliency = binary_mask(imgs[idx].reshape(1, 224, 224, 3),
                                   saliency_map_combined[idx].reshape(1, 224, 224, 3))
        subplot_list = [imgs[idx].reshape(224, 224, 3), point_cloud_saliency[idx].reshape(224, 224, 3),
                        map_saliency.reshape(224, 224, 3)]
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        for ax, title, img in zip(axs, ['Original Image', 'Point Cloud', 'Binary Mask'], subplot_list):
            ax.imshow(img)
        if idx < 5:
            fig.suptitle(["Saliency Map, Correct Classification, Image ID: ", image_net_idx[idx]])
            savefile = "Saliency_Map_Correct_" + str(image_net_idx[idx]) + '.png'
        else:
            fig.suptitle(["Saliency Map, Incorrect Classification, Image ID: ", image_net_idx[idx]])
            savefile = "Saliency_Map_Incorrect_" + str(image_net_idx[idx]) + '.png'
        plt.savefig(savefile)
        plt.show()