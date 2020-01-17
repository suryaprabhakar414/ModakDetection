from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam

#Model Definition
def def_model(pretrained_weights =None):
    input=Input(shape=(224,224,3))
    res=ResNet50(input_tensor=input,include_top=False,pooling='max',weights='imagenet')
    x=Dense(1024,activation='relu')(res.output)
    x=Dropout(rate=0.4)(x)
    out=Dense(2,activation='softmax')(x)
    model=Model(inputs = res.input,outputs = out)
    model.compile(optimizer=Adam(lr=1e-5),loss="categorical_crossentropy",metrics=['accuracy'])
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model


model = def_model(pretrained_weights = None) 
model.summary()
