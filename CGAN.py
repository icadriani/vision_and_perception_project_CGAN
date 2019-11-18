import os
import time

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
from cv2 import imwrite, imread, IMREAD_GRAYSCALE
import tensorflow as tf
import tensorflow_graphics as tfg
import tensorflow_probability as tfp
import tensorboard as tb
import sys

class build_unet_generator():
    """
    Create the U-Net Generator using the hyperparameter values defined below
    """
    def __init__(self,dis):
        self.kernel_size = 4
        self.strides = 2
        self.leakyrelu_alpha = 0.2
        self.upsampling_size = 2
        self.dropout = 0.5
        self.output_channels = 1
        self.input_shape = [1, 256, 256, 1]
        self.discriminator=dis

        self.X=tf.placeholder(tf.float32,shape=self.input_shape)
        self.y=tf.placeholder(tf.float32,shape=self.input_shape)
        
        self.predict=self.model(self.X)
        self.weights=[1E2,1]
        self.loss=tf.reduce_mean(-tf.log(self.discriminator.predict+1E-12))*self.weights[0]+tf.reduce_mean(tf.abs(self.y-self.predict))*self.weights[1]
        self.optimizer=tf.train.AdamOptimizer(learning_rate=6E-5, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss)

    # Encoder Network
    def model(self,data):

        # 1st Convolutional block in the encoder network
        encoder1=tf.layers.conv2d(data,filters=64,kernel_size=self.kernel_size,padding='same',strides=self.strides)
        encoder1=tf.nn.leaky_relu(encoder1,alpha=self.leakyrelu_alpha)

        # 2nd Convolutional block in the encoder network
        encoder2=tf.layers.conv2d(encoder1,filters=128,kernel_size=self.kernel_size,padding='same',strides=self.strides)
        encoder2=tf.layers.batch_normalization(encoder2)
        encoder2=tf.nn.leaky_relu(encoder2,alpha=self.leakyrelu_alpha)

        # 3rd Convolutional block in the encoder network
        encoder3=tf.layers.conv2d(encoder2,filters=256,kernel_size=self.kernel_size,padding='same',strides=self.strides)
        encoder3=tf.layers.batch_normalization(encoder3)
        encoder3=tf.nn.leaky_relu(encoder3,alpha=self.leakyrelu_alpha)

        # 4th Convolutional block in the encoder network
        encoder4=tf.layers.conv2d(encoder3,filters=512,kernel_size=self.kernel_size,padding='same',strides=self.strides)
        encoder4=tf.layers.batch_normalization(encoder4)
        encoder4=tf.nn.leaky_relu(encoder4,alpha=self.leakyrelu_alpha)

        # 5th Convolutional block in the encoder network
        encoder5=tf.layers.conv2d(encoder4,filters=512,kernel_size=self.kernel_size,padding='same',strides=self.strides)
        encoder5=tf.layers.batch_normalization(encoder5)
        encoder5=tf.nn.leaky_relu(encoder5,alpha=self.leakyrelu_alpha)

        # 6th Convolutional block in the encoder network
        encoder6=tf.layers.conv2d(encoder5,filters=512,kernel_size=self.kernel_size,padding='same',strides=self.strides)
        encoder6=tf.layers.batch_normalization(encoder6)
        encoder6=tf.nn.leaky_relu(encoder6,alpha=self.leakyrelu_alpha)

        # 7th Convolutional block in the encoder network
        encoder7=tf.layers.conv2d(encoder6,filters=512,kernel_size=self.kernel_size,padding='same',strides=self.strides)
        encoder7=tf.layers.batch_normalization(encoder7)
        encoder7=tf.nn.leaky_relu(encoder7,alpha=self.leakyrelu_alpha)

        # 8th Convolutional block in the encoder network
        encoder8=tf.layers.conv2d(encoder7,filters=512,kernel_size=self.kernel_size,padding='same',strides=self.strides)
        encoder8=tf.layers.batch_normalization(encoder8)
        encoder8=tf.nn.leaky_relu(encoder8,alpha=self.leakyrelu_alpha)

        # Decoder Network
        # 1st Upsampling Convolutional Block in the decoder network
        decoder1=tfg.image.pyramid.upsample(encoder8,num_levels=1)
        decoder1=tf.layers.conv2d(decoder1[1],filters=512,kernel_size=self.kernel_size,padding='same')
        decoder1=tf.layers.batch_normalization(decoder1)
        decoder1=tf.nn.dropout(decoder1,self.dropout)
        decoder1=tf.concat([decoder1,encoder7],axis=3)
        decoder1=tf.nn.relu(decoder1)

        # 2nd Upsampling Convolutional block in the decoder network
        decoder2=tfg.image.pyramid.upsample(decoder1,num_levels=1)
        decoder2=tf.layers.conv2d(decoder2[1],filters=1024,kernel_size=self.kernel_size,padding='same')
        decoder2=tf.layers.batch_normalization(decoder2)
        decoder2=tf.nn.dropout(decoder2,self.dropout)
        decoder2=tf.concat([decoder2,encoder6],axis=-1)
        decoder2=tf.nn.relu(decoder2)

        # 3rd Upsampling Convolutional block in the decoder network
        decoder3=tfg.image.pyramid.upsample(decoder2,num_levels=1)
        decoder3=tf.layers.conv2d(decoder3[1],filters=1024,kernel_size=self.kernel_size,padding='same')
        decoder3=tf.layers.batch_normalization(decoder3)
        decoder3=tf.nn.dropout(decoder3,self.dropout)
        decoder3=tf.concat([decoder3,encoder5],axis=-1)
        decoder3=tf.nn.relu(decoder3)

        # 4th Upsampling Convolutional block in the decoder network
        decoder4=tfg.image.pyramid.upsample(decoder3,num_levels=1)
        decoder4=tf.layers.conv2d(decoder4[1],filters=1024,kernel_size=self.kernel_size,padding='same')
        decoder4=tf.layers.batch_normalization(decoder4)
        decoder4=tf.concat([decoder4,encoder4],axis=-1)
        decoder4=tf.nn.relu(decoder4)

        # 5th Upsampling Convolutional block in the decoder network
        decoder5=tfg.image.pyramid.upsample(decoder4,num_levels=1)
        decoder5=tf.layers.conv2d(decoder5[1],filters=1024,kernel_size=self.kernel_size,padding='same')
        decoder5=tf.layers.batch_normalization(decoder5)
        decoder5=tf.concat([decoder5,encoder3],axis=-1)
        decoder5=tf.nn.relu(decoder5)

        # 6th Upsampling Convolutional block in the decoder network
        decoder6=tfg.image.pyramid.upsample(decoder5,num_levels=1)
        decoder6=tf.layers.conv2d(decoder6[1],filters=512,kernel_size=self.kernel_size,padding='same')
        decoder6=tf.layers.batch_normalization(decoder6)
        decoder6=tf.concat([decoder6,encoder2],axis=-1)
        decoder6=tf.nn.relu(decoder6)

        # 7th Upsampling Convolutional block in the decoder network
        decoder7=tfg.image.pyramid.upsample(decoder6,num_levels=1)
        decoder7=tf.layers.conv2d(decoder7[1],filters=256,kernel_size=self.kernel_size,padding='same')
        decoder7=tf.layers.batch_normalization(decoder7)
        decoder7=tf.concat([decoder7,encoder1],axis=-1)
        decoder7=tf.nn.relu(decoder7)

        # Last Convolutional layer
        decoder8=tfg.image.pyramid.upsample(decoder7,num_levels=1)
        decoder8=tf.layers.conv2d(decoder8[1],filters=self.output_channels,kernel_size=self.kernel_size,padding='same')
        decoder8=tf.nn.tanh(decoder8)

        return decoder8


class build_patchgan_discriminator():
#     """
#     Create the PatchGAN discriminator using the hyperparameter values defined below
#     """
    def __init__(self):
        self.kernel_size = 4
        self.strides = 2
        self.leakyrelu_alpha = 0.2
        self.padding = 'same'
        self.num_filters_start = 64  # Number of filters to start with
        self.num_kernels = 100
        self.kernel_dim = 5
        self.patchgan_output_dim = (256, 256, 1)
        self.patchgan_patch_dim = [1, 256, 256, 1]
        self.number_patches = int((self.patchgan_output_dim[0] / self.patchgan_patch_dim[1]) * (self.patchgan_output_dim[1] / self.patchgan_patch_dim[2]))
        self.trainable=True

        # Create a list of input layers equal to the number of patches
        self.list_input_layers = [tf.placeholder(shape=self.patchgan_patch_dim,dtype=tf.float32) for _ in range(self.number_patches)]
        self.y=tf.placeholder(tf.float32,shape=(1,2))

        self.logits=self.model(self.list_input_layers)

        self.loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.logits)[0])

        self.optimizer=tf.train.AdamOptimizer(learning_rate=6E-5, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss)

        self.predict=tf.nn.softmax(self.logits)

    def model_patch_gan(self,data):
        des=tf.layers.conv2d(data,filters=64,kernel_size=self.kernel_size,padding=self.padding,strides=self.strides,trainable=self.trainable)
        des=tf.nn.leaky_relu(des,alpha=self.leakyrelu_alpha)

        # Calculate the number of convolutional layers
        total_conv_layers = int(np.floor(np.log(self.patchgan_output_dim[1]) / np.log(2)))
        list_filters = [self.num_filters_start * min(total_conv_layers, (2 ** i)) for i in range(total_conv_layers)]

        # Next 7 Convolutional blocks
        for filters in list_filters[1:]:
            des=tf.layers.conv2d(des,filters=filters,kernel_size=self.kernel_size,padding=self.padding,strides=self.strides,trainable=self.trainable)
            des=tf.layers.batch_normalization(des,trainable=self.trainable)
            des=tf.nn.leaky_relu(des,alpha=self.leakyrelu_alpha)

        # Add a flatten layer
        flatten_layer=tf.layers.flatten(des)

        # Add the final dense layer
        dense_layer=tf.layers.dense(flatten_layer,units=2,activation='softmax',trainable=self.trainable)
        return [dense_layer,flatten_layer]

    def model(self,data):
        # Create the PatchGAN model
        # Pass the patches through the PatchGAN network
        output1 = [self.model_patch_gan(patch)[0] for patch in data]
        output2 = [self.model_patch_gan(patch)[1] for patch in data]

        # In case of multiple patches, concatenate outputs to calculate perceptual loss
        if len(output1) > 1:
            output1=tf.concat(output1,axis=-1)
        else:
            output1 = output1[0]

        # In case of multiple patches, merge output2 as well
        if len(output2) > 1:
            output2=tf.concat(output2,axis=-1)
        else:
            output2 = output2[0]

        # Add a dense layer
        output2 = tf.layers.dense(output2,units=self.num_kernels * self.kernel_dim, use_bias=False, activation=None,trainable=self.trainable)

        # Add a lambda layer
        output2 = tf.exp(-tf.reduce_sum(tf.abs(tf.expand_dims(output2, -1)  - tf.expand_dims(output2, 0)), 2))

        # Finally concatenate output1 and output2
        output1 = tf.concat([output1, output2],axis=-1)
        final_output = tf.layers.dense(output1,2,trainable=self.trainable)

        return final_output



"""
Data preprocessing methods
"""


def generate_and_extract_patches(images, datas, generator_model, batch_counter, patch_dim, sess):
    # Alternatively, train the discriminator network on real and generated images
    if batch_counter % 2 == 0:
        # Generate fake images
        feed_dict={generator_model.X:datas,generator_model.y:images}
        output_images=sess.run(generator_model.predict,feed_dict=feed_dict)

        # Create a batch of ground truth labels
        labels = np.zeros((output_images.shape[0], 2), dtype=np.uint8)
        labels[:, 0] = 1

    else:
        # Take real images
        output_images = images

        # Create a batch of ground truth labels
        labels = np.zeros((output_images.shape[0], 2), dtype=np.uint8)
        labels[:, 1] = 1

    patches = []
    for y in range(0, output_images.shape[0], patch_dim[0]):
        for x in range(0, output_images.shape[1], patch_dim[1]):
            image_patches = output_images[:, y: y + patch_dim[0], x: x + patch_dim[1], :]
            patches.append(np.asarray(image_patches, dtype=np.float32))

    return patches, labels


def save_images(real_images, real_sketches, generated_images, num_epoch, dataset_name, limit):
    real_sketches = real_sketches * 255.0
    real_images = real_images * 255.0
    generated_images = generated_images * 255.0

    # Save stack of images
    imwrite('results/X_full_real_image_{}_{}.png'.format(dataset_name, num_epoch), real_images)
    imwrite('results/X_full_real_sketch_{}_{}.png'.format(dataset_name, num_epoch), real_sketches)
    imwrite('results/X_full_generated_image_{}_{}.png'.format(dataset_name, num_epoch), generated_images)


def load_dataset(data_dir, data_type, img_width, img_height):
    data_dir_path = os.path.join(data_dir, data_type)

    if not os.path.exists(data_dir_path+'A'): 
        return [],[]

    data_photos_jpg = [os.path.join(data_dir_path+'A',f) for f in os.listdir(data_dir_path+'A') if '.jpg' in f]
    data_labels_jpg = [os.path.join(data_dir_path+'B',f) for f in os.listdir(data_dir_path+'B') if '.jpg' in f]

    final_data_photos = None
    final_data_labels = None
    data_photos = []
    data_labels = []

    for index in range(len(data_photos_jpg)):
        print('Loading dataset ',data_type,'... ',round(index*100/len(data_photos_jpg)),'%, ',index+1,'/',len(data_photos_jpg),sep='',end='\r')

        data_photo=imread(data_photos_jpg[index],IMREAD_GRAYSCALE)
        data_label=imread(data_labels_jpg[index],IMREAD_GRAYSCALE)
        
        data_photos.append(data_photo)
        data_labels.append(data_label)

    # Resize and normalize images

    all_datas_photos = np.array(data_photos, dtype=np.float32)
    num_photos = all_datas_photos.shape[0]
    all_datas_photos = all_datas_photos.reshape((num_photos, img_width, img_height, 1)) / 255.0

    all_datas_labels = np.array(data_labels, dtype=np.float32)
    num_labels = all_datas_labels.shape[0]
    all_datas_labels = all_datas_labels.reshape((num_labels, img_width, img_height, 1)) / 255.0

    if final_data_photos is not None and final_data_labels is not None:
        final_data_photos = np.concatenate([final_data_photos, all_datas_photos], axis=0)
        final_data_labels = np.concatenate([final_data_labels, all_datas_labels], axis=0)
    else:
        final_data_photos = all_datas_photos
        final_data_labels = all_datas_labels

    return final_data_photos, final_data_labels

def splittingValTest(val,val_labels,test,test_labels):
    if len(val)>0 and len(test)>0: return
    new_val=None
    new_val_labels=None
    new_test=None
    new_test_labels=None
    if len(val)==0:
        p=round(len(test)/4)
        new_val=test[:p]
        new_val_labels=test_labels[:p]
        new_test=test[p:]
        new_test_labels=test_labels[p:]
    else:
        p=round(len(val)/3)
        new_test=val[:p]
        new_test_labels=val_labels[:p]
        new_val=val[p:]
        new_val_labels=val_labels[p:]
    return new_val,new_val_labels,new_test,new_test_labels



def write_log(callback, name, loss, batch_no):
    """
    Write training summary to TensorBoard
    """
    tf.summary.scalar(name,loss)
    tf.summary.scalar(name,batch_no)
    callback.flush()


if __name__ == '__main__':
    epochs = 10 
    num_images_per_epoch = 400
    batch_size = 1
    img_width = 256
    img_height = 256
    num_channels = 1
    input_img_dim = (256, 256, 1)
    patch_dim = (256, 256)
    dataset_dir = "data\\"+sys.argv[1]+"\\"

    """
    # Build and compile networks
    # """
    print('Building discriminator network...')
    patchgan_discriminator = build_patchgan_discriminator()

    print('Building fake discriminator network...')
    patchgan_discriminator_fake = build_patchgan_discriminator()

    print('Building generator network...')
    unet_generator = build_unet_generator(patchgan_discriminator_fake)

    """
    Load the training, testing and validation datasets
    """
    training_data_photos, training_data_labels = load_dataset(data_dir=dataset_dir, data_type='train',img_width=img_width, img_height=img_height)

    test_data_photos, test_data_labels = load_dataset(data_dir=dataset_dir, data_type='test', img_width=img_width, img_height=img_height)

    validation_data_photos, validation_data_labels = load_dataset(data_dir=dataset_dir, data_type='val', img_width=img_width, img_height=img_height)

    validation_data_photos, validation_data_labels, test_data_photos, test_data_labels=splittingValTest(validation_data_photos, validation_data_labels, test_data_photos, test_data_labels)

    training_data_photos=training_data_photos[:1000]
    training_data_labels=training_data_photos[:1000]
    validation_data_photos=validation_data_photos[:50]
    validation_data_labels=validation_data_photos[:50]
    test_data_photos=test_data_photos[:300]
    test_data_labels=test_data_labels[:300]

    with tf.Session() as sess:
        tensorboard = tf.summary.FileWriter("logs/{}".format(time.time()),sess.graph)
        saver_gen=tf.train.Saver()
        saver_dis=tf.train.Saver()

        print('Inizializing...')
        sess.run(tf.global_variables_initializer())

        tot_dis_losses=[]
        tot_gen_losses=[]

        print('Starting the training...')
        for epoch in range(0, epochs):
            print('Epoch {}'.format(epoch))

            dis_losses = []
            gen_losses = []

            batch_counter = 1
            start = time.time()

            num_batches = int(training_data_photos.shape[0] / batch_size)

            # Train the networks for number of batches
            for index in range(int(training_data_photos.shape[0] / batch_size)):
                print("Batch:{}".format(index))

                # Sample a batch of training and validation images
                train_datas_batch = training_data_labels[index * batch_size:(index + 1) * batch_size]
                train_images_batch = training_data_photos[index * batch_size:(index + 1) * batch_size]

                val_datas_batch = validation_data_labels[index * batch_size:(index + 1) * batch_size]
                val_images_batch = validation_data_photos[index * batch_size:(index + 1) * batch_size]

                patches, labels = generate_and_extract_patches(train_images_batch, train_datas_batch, unet_generator, batch_counter, patch_dim, sess)
                
                labels_adv = np.zeros((train_images_batch.shape[0], 2), dtype=np.float32)
                labels_adv[:, 1] = 1.0

                labels_fake = np.zeros((train_images_batch.shape[0], 2), dtype=np.float32)

                """
                Train the discriminator model
                """

                for i in range(len(patches)):
                    feed_dict={patchgan_discriminator.y:np.expand_dims(labels[i],axis=0),
                               unet_generator.X:np.expand_dims(train_datas_batch[i],0),
                               unet_generator.y:np.expand_dims(train_images_batch[i],0)}
                    for j in range(len(patchgan_discriminator.list_input_layers)):
                        feed_dict[patchgan_discriminator.list_input_layers[j]]=np.expand_dims(patches[i][j],axis=0)
                    generated_images=sess.run(unet_generator.predict,feed_dict=feed_dict)
                    feed_dict[patchgan_discriminator_fake.y]=np.expand_dims(labels[i],0)
                    for j in range(len(patchgan_discriminator_fake.list_input_layers)):
                        feed_dict[patchgan_discriminator_fake.list_input_layers[j]]=np.expand_dims(generated_images[i],axis=0)
                    d_loss,_,g_loss,_=sess.run([patchgan_discriminator.loss,patchgan_discriminator.optimizer,
                                                      unet_generator.loss,unet_generator.optimizer],feed_dict=feed_dict)


                # Increase the batch counter
                batch_counter += 1

                print("Discriminator loss:", d_loss)
                print("Generator loss:", g_loss)

                gen_losses.append(g_loss)
                dis_losses.append(d_loss)

            tot_dis_losses.append(np.mean(dis_losses))
            tot_gen_losses.append(np.mean(gen_losses))

            """
            Save losses to Tensorboard after each epoch
            """
            write_log(tensorboard, 'discriminator_loss', np.mean(dis_losses), epoch)
            write_log(tensorboard, 'generator_loss', np.mean(gen_losses), epoch)

            # After every 10th epoch adn on last epoch, generate and save images for visualization
            if epoch % 10 == 0 or epoch==epochs-1:
                
                for i in range(len(validation_data_labels)):
                    print('Validation... ',round(i*100/len(validation_data_labels)),'%',sep='',end='\r')
                    feed_dict={unet_generator.X:np.expand_dims(validation_data_labels[i],0),unet_generator.y:np.expand_dims(validation_data_photos[i],0)}
                    # Generate images
                    validation_generated_images=sess.run(unet_generator.predict,feed_dict=feed_dict)[0]

                    # Save images
                    save_images(validation_data_photos[i], validation_data_labels[i], validation_generated_images, epoch, 'validation', limit=5)

        for i in range(len(test_data_photos)):
          print('Testing... ',round(i*100/len(test_data_photos)),'%, ',i+1,'/',len(test_data_photos),sep='',end='\r')
          feed_dict={unet_generator.X:np.expand_dims(test_data_labels[i],0),unet_generator.y:np.expand_dims(test_data_photos[i],0)}
          generated_image=sess.run(unet_generator.predict,feed_dict=feed_dict)[0]
          save_images(test_data_labels[i], test_data_photos[i], generated_image, 0, 'test', limit=5)
        print()

        plt.figure()
        plt.plot(tot_dis_losses,'r')
        plt.savefig('dis_losses.png')

        plt.figure()
        plt.plot(tot_gen_losses,'b')
        plt.savefig('gen_losses.png')
        

        print('Saving the generator and the discriminator')
        saver_gen.save(sess,'generator/generator.ckpt')
        saver_dis.save(sess,'discriminator/discriminator.ckpt')



