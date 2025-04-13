# utils/data_pipeline.py

import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_dir, img_height=224, img_width=224, batch_size=32, validation_split=0.2):
    """
    Creates and returns training and validation data generators.

    Args:
        data_dir (str): Path to the dataset (expects subdirectories per class).
        img_height (int): Target image height.
        img_width (int): Target image width.
        batch_size (int): Number of images per batch.
        validation_split (float): Fraction of data reserved for validation.

    Returns:
        train_generator, validation_generator: Keras ImageDataGenerator objects.
    """
    # Define an ImageDataGenerator with data augmentation for the training set.
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,          # Random rotations up to 20 degrees.
        width_shift_range=0.1,      # Horizontal shift.
        height_shift_range=0.1,     # Vertical shift.
        shear_range=0.1,            # Shearing transformations.
        zoom_range=0.1,             # Random zoom.
        horizontal_flip=True,       # Random horizontal flip.
        fill_mode='nearest',        # Fill mode for new pixels.
        validation_split=validation_split  # Reserve part of data for validation.
    )

    # Set up the training generator
    train_generator = datagen.flow_from_directory(
        directory=data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',  # Uses (1 - validation_split) fraction of images.
        shuffle=True
    )

    # Set up the validation generator
    validation_generator = datagen.flow_from_directory(
        directory=data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',  # Uses validation_split fraction of images.
        shuffle=False
    )

    return train_generator, validation_generator

def visualize_batch(generator, num_images=5):
    """
    Visualizes a batch of images from the generator.

    Args:
        generator: An instance of a Keras ImageDataGenerator.
        num_images: Number of images to display.
    """
    images, labels = next(generator)
    plt.figure(figsize=(15, 5))
    for i in range(min(num_images, len(images))):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Construct the path to your dataset (adjust this as needed)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'datasets', 'raw')
    print("Loading data from:", data_dir)

    # Retrieve the training and validation generators
    train_gen, val_gen = get_data_generators(data_dir)

    # Print sample counts for verification
    print("Training samples:", train_gen.samples)
    print("Validation samples:", val_gen.samples)

    # Visualize one batch from the training set
    visualize_batch(train_gen)