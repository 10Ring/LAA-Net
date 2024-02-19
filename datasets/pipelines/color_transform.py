import os
import sys
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

from datasets.builder import PIPELINES
import albumentations as A


@PIPELINES.register_module()
class ColorJitterTransform(object):
    def __init__(self,
                 clahe: float,
                 colorjitter: float,
                 gaussianblur: float,
                 jpegcompression: list,
                 rgbshift: float,
                 randomcontrast: float,
                 randomgamma: float,
                 randombrightness: float,
                 huesat: float,
                 gaussnoise: float,
                 *args,
                 **kwargs):
        super().__init__()
        self.clahe = clahe
        self.colorjitter = colorjitter
        self.gaussianblur = gaussianblur
        self.jpegcompression = jpegcompression
        self.rgbshift = rgbshift
        self.randomcontrast = randomcontrast
        self.randomgamma = randomgamma
        self.randombrightness = randombrightness
        self.huesat = huesat
        self.gaussnoise = gaussnoise
        
        if kwargs is not None:
            for k,v in kwargs.items():
                if v is None:
                    raise ValueError(f'{k}:{v} retrieve a None value!')
                self.__setattr__(k, v)

    def _CLAHE(self, 
               clip_limit=4.0, 
               tile_grid_size=(8, 8), 
               always_apply=False, 
               p=0.5):
        return A.CLAHE(clip_limit=clip_limit, 
                       tile_grid_size=tile_grid_size, 
                       always_apply=always_apply, 
                       p=p)
    
    def _colorjitter(self, 
                     brightness=0.2, 
                     contrast=0.2, 
                     saturation=0.2, 
                     hue=0.2, 
                     always_apply=False, 
                     p=0.5):
        return A.ColorJitter(brightness=brightness, 
                             contrast=contrast, 
                             saturation=saturation, 
                             hue=hue, 
                             always_apply=always_apply, 
                             p=p)
    
    def _gaussianblur(self, 
                      blur_limit=(3, 7), 
                      sigma_limit=0, 
                      always_apply=False, 
                      p=0.5):
        return A.GaussianBlur(blur_limit=blur_limit, 
                              sigma_limit=sigma_limit, 
                              always_apply=always_apply, 
                              p=p)
    
    def _gauss_noise(self,
                     var_limit=(10.0, 50.0), 
                     mean=0, 
                     per_channel=True, 
                     always_apply=False, 
                     p=0.5):
        return A.GaussNoise(var_limit=var_limit,
                            mean=mean,
                            per_channel=per_channel,
                            always_apply=always_apply,
                            p=p)
    
    def _jpegcompression(self, 
                         quality_lower=70, 
                         quality_upper=100, 
                         always_apply=False, 
                         p=0.5):
        return A.ImageCompression(quality_lower=quality_lower, 
                                  quality_upper=quality_upper, 
                                  always_apply=always_apply, 
                                  p=p)

    def _rgbshift(self,
                  r_shift_limit=20, 
                  g_shift_limit=20, 
                  b_shift_limit=20, 
                  always_apply=False, 
                  p=0.5):
        return A.RGBShift(r_shift_limit=r_shift_limit, 
                          g_shift_limit=g_shift_limit, 
                          b_shift_limit=b_shift_limit, 
                          always_apply=always_apply, 
                          p=p)

    def _randomcontrast(self, 
                        limit=0.2, 
                        always_apply=False, 
                        p=0.5):
        return A.RandomContrast(limit=limit, 
                                always_apply=always_apply, 
                                p=p)
    
    def _randombrightness(self, 
                          brightness_limit=0.1, 
                          contrast_limit=0.1, 
                          brightness_by_max=True, 
                          always_apply=False, 
                          p=0.5):
        return A.RandomBrightnessContrast(brightness_limit=brightness_limit,
                                          contrast_limit=contrast_limit,
                                          brightness_by_max=brightness_by_max,
                                          always_apply=always_apply,
                                          p=p)
    
    def _randomgamma(self, 
                     gamma_limit=(80, 120), 
                     eps=None, 
                     always_apply=False, 
                     p=0.5):
        return A.RandomGamma(gamma_limit=gamma_limit,
                             eps=eps,
                             always_apply=always_apply,
                             p=p)
    
    def _huesaturation(self, 
                       hue_shift_limit=20, 
                       sat_shift_limit=20, 
                       val_shift_limit=20, 
                       always_apply=False, 
                       p=0.5):
        return A.HueSaturationValue(hue_shift_limit=hue_shift_limit,
                                    sat_shift_limit=sat_shift_limit,
                                    val_shift_limit=val_shift_limit,
                                    always_apply=always_apply,
                                    p=p)
    
    def __call__(self, x):
        transforms = [
            A.Compose([
                self._CLAHE(p=self.clahe),
                self._randomcontrast(p=self.randomcontrast),
                self._colorjitter(p=self.colorjitter),
                self._jpegcompression(p=self.jpegcompression[0],
                                      quality_lower=self.jpegcompression[1],
                                      quality_upper=self.jpegcompression[2]),
                self._rgbshift(p=self.rgbshift),
                self._randomgamma(p=self.randomgamma),
            ]),
            A.OneOf([
                self._gaussianblur(p=self.gaussianblur),
                self._gauss_noise(p=self.gaussnoise),
            ]),
            A.OneOf([
                self._randombrightness(p=self.randombrightness),
                self._huesaturation(p=self.huesat)
            ])
        ]
        return A.Compose(transforms)(image=x)
