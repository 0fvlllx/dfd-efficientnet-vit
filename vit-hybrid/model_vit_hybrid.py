import timm

def vit_base_r50_s16_224_in21k(**kwargs):
    model = timm.create_model('vit_base_r50_s16_224_in21k',
                              pretrained=True, **kwargs)
    return model

vit_base_r50_s16_224 = vit_base_r50_s16_224_in21k


