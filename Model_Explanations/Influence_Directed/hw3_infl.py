import numpy as np
import tensorflow as tf

class InflDirected(object):
    """Implementaion of Influence Directed Explanation

        https://arxiv.org/abs/1802.03788
    """
    def __init__(self, sess,
                 input_tensor, internal_layer,
                 output_tensor, qoi):
        """__Constructor__

        Arguments:
            sess {tf.Session} -- Tensorflow session
            input_tensor {tf.Tensor} -- Symbolic tensor of (batch of) inputs
            internal_layer {tf.Tensor} -- Symbolic tensor of internal layer
              output
            output_tensor {tf.Tensor} -- Symbolic tensor of (batch of)
              pre-softmax outputs
            qoi {tf.Tensor -> tf.Tensor} -- The quantity of interest, a
              function that takes in a single instance output and produces
              the quantity of interest tensor for that output.

            Example usage of qoi:
              InflDirected(sess, input_batch, layer,
                           output_batch, qoi=lambda out: out[target_class])
            # for some target class target_class

        """
        self.sess = sess
        self.input_tensor = input_tensor
        self.internal_layer = internal_layer
        self.output_tensor = output_tensor

        self.qoi = tf.map_fn(qoi, self.output_tensor)

        self._define_ops()

    def _define_ops(self):
        """Add whatever operations you feel necessary into the computational
        graph. Feel free to add more helper functions in you feel this one is
        not enough for your implementation
        """

        # >>> Your code starts here <<<
        y_s = self.qoi
        x_s = self.internal_layer
        self.gradients = tf.gradients(ys=y_s, xs=x_s)[0]  #generate symbolic tensor for gradient calculation
        # >>> Your code ends here <<<

    def dis_influence(self, X, batch_size=16):
        """Compute the distribution of influence and return the expert neuron

        Arguments:
            X {np.ndarray} -- Input dataset

        Keyword Arguments:
            batch_size {int} -- Batch Size (default: {16})

        Returns:
            int -- The expert neuron in the internal layer.
        """
        expert_id = None
        # >>> Your code starts here <<<
        num_images = X.shape[0] #get number of images
        gradients = self.sess.run([self.gradients], feed_dict={self.input_tensor: X}) #calculate the gradients with X being fed in
        gradients = np.asarray(gradients) / num_images
        expert_id = np.argmax(gradients) #take argmax to locate expert nueron
        # >>> Your code ends here <<<
        return expert_id

    def expert_attribution(self,
                           expert_id,
                           X,
                           batch_size=16,
                           multiply_with_input=True):
        """__call__ forward computation to generate the saliency map

        Arguments:
            expert_id {int} -- The nueron index of expert
            X {np.ndarray} -- Input dataset

        Keyword Arguments:
            batch_size {int} -- Batch size (default: {16})
            multiply_with_input {bool} -- If True, return grad x input,
                else return grad (default: {True})

        Returns np.ndarray of the same shape as X.
        """
        # #>>> Your code starts here <<<
        y_s = self.internal_layer[:, expert_id]   #symbolic tensor for generating the output of the internal layer at the expert nueron
        x_s = self.input_tensor
        self.saliency_grads = tf.gradients(ys=y_s, xs=x_s)[0] #symbolic tensor to calculate the gradients
        self.saliency_output_grads = tf.multiply(self.input_tensor, self.saliency_grads)  #symbolic tensor to calculate gradients multiplied with input
        gradients, multiplied_gradients = self.sess.run([self.saliency_grads, self.saliency_output_grads], feed_dict={self.input_tensor: X})
        if multiply_with_input:
            return multiplied_gradients
        else:
            return gradients


if __name__ == '__main__':
    # >>> Your code for written exercises here <<<
    image_set = load_ImageNet()   #loading the VGG data
    weasel_indexes = []  #creation of empty lists to hold indexes for the classes needed/chosen
    camel_indexes = []
    persian_indexes = []
    siamese_indexes = []
    ''' Go through the ground truth labels and append the index corresponding to it to the correct list'''
    for idx, label in enumerate(image_set[1]):
        if label == 356:
            weasel_indexes.append(idx)
        if label == 354:
            camel_indexes.append(idx)
        if label == 283:
            persian_indexes.append(idx)
        if label == 284:
            siamese_indexes.append(idx)
    #creation of image subsets and all images as arrays to be used later for classes chosen
    weasel_images = np.zeros((len(weasel_indexes), 224, 224, 3))
    weasel_images = np.asarray([image_set[0][idx] for idx in weasel_indexes]).astype(float)
    camel_images = np.zeros((len(camel_indexes), 224, 224, 3))
    camel_images = np.asarray([image_set[0][idx] for idx in camel_indexes]).astype(float)
    persian_cat = np.zeros((len(persian_indexes), 224, 224, 3))
    persian_cat = np.asarray([image_set[0][idx] for idx in persian_indexes]).astype(float)
    siamese_cat = np.zeros((len(siamese_indexes), 224, 224, 3))
    siamese_cat = np.asarray([image_set[0][idx] for idx in siamese_indexes]).astype(float)
    all_images = np.zeros((len(image_set[0]), 224, 224, 3))
    all_images = np.asarray([image_set[0][idx] for idx in range(len(image_set[0]))]).astype(float)

    '''The code below was used to generate the attribution maps to be viewed in order to select images to visualize in my report'''
    target = ImageNet_name2idx("Arabian camel, dromedary, Camelus dromedarius")  #target label id from the function given
    with tf.Session() as session:
        model = get_model(session)  #get VGG model
        InflD = InflDirected(session, model.imgs, model.fc1, model.fc3l, lambda x: x[target])  #initialize Influence Direct Object for specified target class
        expert_unit = InflD.dis_influence(image_set[0], 1)  #find the expert unit corresponding to the distribution of influence
        print('Expert Neuron for Camel Target for all mages: ', expert_unit)
        attr_map_camel_all = InflD.expert_attribution(expert_unit, all_images)  #get the attribution map for all images corresponding to the expert unit
        expert_unit_camel = InflD.dis_influence(camel_images, 1)   #find the expert unit for subset
        print('Expert Neuron for Camel Target for Camel images: ', expert_unit_camel)
        attr_map_camel_only = InflD.expert_attribution(expert_unit_camel, camel_images)  #get attribution map for subset corresponding to its expert unit

    target = ImageNet_name2idx("weasel")  #target label id from the function given
    with tf.Session() as session:
        model = get_model(session) #get VGG model
        InflD = InflDirected(session, model.imgs, model.fc1, model.fc3l, lambda x: x[target]) #initialize Influence Direct Object for specified target class
        expert_unit = InflD.dis_influence(image_set[0], 1) #find the expert unit corresponding to the distribution of influence
        print('Expert Neuron for Weasel Target for all images: ', expert_unit)
        attr_map_weasel_all = InflD.expert_attribution(expert_unit, all_images) #get the attribution map for all images corresponding to the expert unit
        expert_unit_weasel = InflD.dis_influence(weasel_images, 1) #find the expert unit for subset
        print('Expert Neuron for Weasel Target for Weasel images: ', expert_unit_weasel)
        attr_map_weasel_only = InflD.expert_attribution(expert_unit_weasel, weasel_images) #get attribution map for subset corresponding to its expert unit

    label = ImageNet_name2idx("Persian cat")  #target label id from the function given
    print(label)
    with tf.Session() as session:
        model = get_model(session) #get VGG model
        InflD = InflDirected(session, model.imgs, model.fc1, model.fc3l, lambda x: x[label]) #initialize Influence Direct Object for specified target class
        expert_unit = InflD.dis_influence(image_set[0], 1) #find the expert unit corresponding to the distribution of influence
        print('Expert Neuron for Persian Cat Target for all images: ', expert_unit)
        attr_map_persian_all = InflD.expert_attribution(expert_unit, all_images) #get the attribution map for all images corresponding to the expert unit
        expert_unit_persian = InflD.dis_influence(persian_cat, 1) #find the expert unit for subset
        print('Expert Neuron for Persian Cat Target for Persian Cat images: ', expert_unit_persian)
        attr_map_persian_only = InflD.expert_attribution(expert_unit_persian, persian_cat) #get attribution map for subset corresponding to its expert unit

    target = ImageNet_name2idx("Siamese cat, Siamese")  #target label id from the function given
    with tf.Session() as session:
        model = get_model(session) #get VGG model
        InflD = InflDirected(session, model.imgs, model.fc1, model.fc3l, lambda x: x[target]) #initialize Influence Direct Object for specified target class
        expert_unit = InflD.dis_influence(image_set[0], 1) #find the expert unit corresponding to the distribution of influence
        print('Expert Neuron for Siamese Cat Target for all images:: ', expert_unit)
        attr_map_siamese_all = InflD.expert_attribution(expert_unit, all_images) #get the attribution map for all images corresponding to the expert unit
        expert_unit_siamese = InflD.dis_influence(siamese_cat, 1) #find the expert unit for subset
        print('Expert Neuron for Siamese Cat Target for Siamese Cat images: ', expert_unit_siamese)
        attr_map_siamese_only = InflD.expert_attribution(expert_unit_siamese, siamese_cat) #get attribution map for subset corresponding to its expert unit

    target = ImageNet_name2idx("Siamese cat, Siamese") - ImageNet_name2idx("Persian cat")  #difference in target label ids from the function given
    with tf.Session() as session:
        model = get_model(session) #get VGG model
        InflD = InflDirected(session, model.imgs, model.fc1, model.fc3l, lambda x: x[target]) #initialize Influence Direct Object for specified target class
        expert_unit = InflD.dis_influence(image_set[0], 1) #find the expert unit corresponding to the distribution of influence
        print('Expert Neuron for Difference in Cats in all images: ', expert_unit)
        attr_map_difference_all = InflD.expert_attribution(expert_unit, all_images) #get the attribution map for all images corresponding to the expert unit
        expert_unit_siamese = InflD.dis_influence(siamese_cat, 1) #find the expert unit for subset
        print('Expert Neuron for Difference in Cats in Siamese Cat images: ', expert_unit_siamese)
        attr_map_siamese = InflD.expert_attribution(expert_unit_siamese, siamese_cat) #get attribution map for subset corresponding to its expert unit


    '''After each run of a session to get the DOI for whole dataset and subset of target class for designated class, I would
       use a simple little search in order to visualize the attribution maps and determine which images to show in my report.
       I would just manually change out the attribution map specified and indices for start and finish depending on the class being 
       viewed.'''
    '''This would be to view image attribution maps after running the Influence Map over the entire dataset as the DOI.'''
    idx_start = 800
    idx_finish = 900
    for index, img in enumerate(all_images[idx_start:idx_finish]):
        print(index + idx_start)
        map = binary_mask(img.reshape(1, 224, 224, 3), attr_map_all[index + idx_start].reshape(1, 224, 224, 3)) #create binary map for visualization
        plt.imshow(map.reshape(224, 224, 3))
        plt.show()
    '''Example of looking at a specific class images where DOI is the class subset. Would manually change it. Or setup separate blocks in Colab'''
    idx_start = 900 #indicate the start of class index
    for index, img in enumerate(camel_images):
        print(index + idx_start)
        map = binary_mask(img.reshape(1, 224, 224, 3), attr_map_camel_only[index].reshape(1, 224, 224, 3))
        plt.imshow(map.reshape(224, 224, 3))
        plt.show()

    '''After determining which indices I wanted to use, I created these functions for visualizing and saving the images as needed.'''
    def visualization(index, attr_map, all_images_set):
        visual_map = binary_mask(all_images[index].reshape(1, 224, 224, 3), attr_map[index].reshape(1, 224, 224, 3)) #create visualization
        pci = point_cloud(attr_map[index]) #create point cloud
        sub_plot_list = [all_images_set[index].reshape(224, 224, 3).astype(int), pci, visual_map.reshape(224, 224, 3)]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for ax, title, img in zip(axes, ['Original Input ', 'Expert at fc_1 ', 'Visualization '], sub_plot_list): #create subplots
            ax.imshow(img)
            ax.set_title(title)
        file_to_save = "Difference_Cat_Target_Whole_Dist_" + str(index) + '.png'
        plt.savefig(file_to_save)
        plt.show()

    '''This visualization tool was to use for subsets as DOI'''
    def visualization_subset(index, attr_map, class_images):
        visual_map = binary_mask(class_images[index].reshape(1, 224, 224, 3), attr_map[index].reshape(1, 224, 224, 3))
        pci = point_cloud(attr_map[index])
        sub_plot_list = [class_images[index].reshape(224, 224, 3).astype(int), pci, visual_map.reshape(224, 224, 3)]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for ax, title, img in zip(axes, ['Original Input ', 'Expert at fc_1 ', 'Visualization '], sub_plot_list):
            ax.imshow(img)
            ax.set_title(title)
        file_to_save = "Difference_Cat_Target_Siamese_Cat_Dist_" + str(index) + '.png'
        plt.savefig(file_to_save)
        plt.show()


    '''The following code was used to generate all visuals for this portion of the homework after determining proper indices
        of images I wanted to use.'''
    weasel_plot_all = [18, 384, 887]
    for idx in weasel_plot_all:
        visualization(idx, attr_map_weasel_all, all_images)
    weasel_plot_weasel_only = [99, 48]
    for idx in weasel_plot_weasel_only:
        visualization_subset(idx, attr_map_weasel_only, weasel_images)

    camel_plot_all = [525, 902, 622]
    for idx in camel_plot_all:
        visualization(idx, attr_map_camel_all, all_images)
    camel_plot_subset = [86, 81, 79]
    for idx in camel_plot_subset:
        visualization_subset(idx, attr_map_camel_only, camel_images)

    persian_plot_all = [689, 828, 754, 887]
    for idx in persian_plot_all:
        visualization(idx, attr_map_persian_all, all_images)
    persian_plot_subset = [97, 86, 89, 84, 83, 78, 74]
    for idx in persian_plot_subset:
        visualization_subset(idx, attr_map_persian_only, persian_cat)

    siamese_plot_all = [790, 180, 161, 130, 979]
    for idx in siamese_plot_all:
        visualization(idx, attr_map_siamese_all, all_images)
    siamese_plot_subset = [62, 60, 38]
    for idx in siamese_plot_subset:
        visualization_subset(idx, attr_map_siamese_only, siamese_cat)

    difference_plot_all = [797, 786, 771, 98, 62, 541, 525, 514]
    for idx in difference_plot_all:
        visualization(idx, attr_map_difference_all, all_images)
    difference_plot_subset = [28, 21, 33, 24]
    for idx in difference_plot_subset:
        visualization_subset(idx, attr_map_siamese, siamese_cat)

    pass
