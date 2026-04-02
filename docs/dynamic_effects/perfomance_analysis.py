from main import main

tests = {
    "glasses only": dict(ENABLE_GLASSES=True, ENABLE_HAT=False, ENABLE_SMOOTH=False, 
                           ENABLE_WHITENING=False, ENABLE_LIPSTICK=False, frames_num=300),
    "hat only": dict(ENABLE_GLASSES=False, ENABLE_HAT=True, ENABLE_SMOOTH=False, 
                         ENABLE_WHITENING=False, ENABLE_LIPSTICK=False, frames_num=300),
    "smooth only": dict(ENABLE_GLASSES=False, ENABLE_HAT=False, ENABLE_SMOOTH=True, 
                        ENABLE_WHITENING=False, ENABLE_LIPSTICK=False, frames_num=300),                         
    "whitening only": dict(ENABLE_GLASSES=False, ENABLE_HAT=False, ENABLE_SMOOTH=False, 
                        ENABLE_WHITENING=True, ENABLE_LIPSTICK=False, frames_num=300),
    "lipstick only": dict(ENABLE_GLASSES=False, ENABLE_HAT=False, ENABLE_SMOOTH=False, 
                        ENABLE_WHITENING=False, ENABLE_LIPSTICK=True, frames_num=300),
    "sticker only": dict(ENABLE_GLASSES=True, ENABLE_HAT=True, ENABLE_SMOOTH=False, 
                        ENABLE_WHITENING=False, ENABLE_LIPSTICK=False, frames_num=300),
    "beauty only": dict(ENABLE_GLASSES=False, ENABLE_HAT=False, ENABLE_SMOOTH=True, 
                        ENABLE_WHITENING=True, ENABLE_LIPSTICK=True, frames_num=300),
    "both stickers and beauty effects": dict(ENABLE_GLASSES=True, ENABLE_HAT=True, ENABLE_SMOOTH=True, 
                        ENABLE_WHITENING=True, ENABLE_LIPSTICK=True, frames_num=300),
}

for name, config in tests.items():
    fps = main(**config)
    print(f"{name}: {fps:.2f} FPS")