from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50, VGG16



class ClassificationModelMaker:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape  # インスタンス変数を定義(ここで指定した変数は、class内ならどこからでもアクセス可能)
        self.num_classes = num_classes
        pass

    def alexnet(self):
        model = Sequential()
        model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=self.input_shape, padding="valid"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
        model.add(Conv2D(256, (5, 5), strides=(1, 1), activation='relu', padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
        model.add(Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"))
        model.add(Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding="same"))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='softmax'))
        return model

    def resnet50(self):
        base_model = ResNet50(weights=None, include_top=False, input_shape=self.input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def vgg16(self):
        base_model = VGG16(weights=None, include_top=False, input_shape=self.input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model