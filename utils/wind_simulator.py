
# from here: https://en.ilmatieteenlaitos.fi/download-observations 2020,2,10,02:00,UTC onward
# gust speed first, then windspeed

# TODO
# Make wind something that comes from a distribtuion and set a landing time variable
# if wind goes over threshold landing must be aborted

# 1) tuulen nopeus tulee jostakin jatkuva-arvoisesta funktiosta, kuten sinikäyrä tmv.
# 2) jokaisessa trialissa arvotaan skenaarion parametrit, jolloin tuulen nopeus samplataan jostakin jakaumasta, ja tuon jakauman observoitu hajonta on stressi;
# 3) odotettu negatiivinen reward on stressitaso.. tässä vain ideoita


# knots = 38 (dry or wet) knots = 25 (snow) knots = 20 (3mm water) knots = 15 (ice)
# crosswind = sivutuuli
# laskeutumista yritetään jostain korkeudesta?
# windspeed from here: https://weatherspark.com/h/d/80053/2021/1/22/Historical-Weather-on-Friday-January-22-2021-in-Linköping-Sweden#Figures-WindSpeed
# 10. helmikuuta 2021 turussa tuulinen päivä

# from here: https://en.ilmatieteenlaitos.fi/download-observations 2020,2,10,02:00,UTC onward
# gust speed first, then windspeed

# %%
weather_data = [
    (20.6,9.2),
    (20.6,8.8),
    (20.6,9),
    (20.6,9.6),
    (20.6,9.2),
    (20.6,9.7),
    (20.6,9.6),
    (20.6,9.5),
    (20.6,9.5),
    (20.6,9.9),
    (17.8,9.7),
    (17.8,9.2),
    (17.8,9.3),
    (17.8,9.1),
    (20.4,9.3),
    (20.6,9.8),
    (20.6,10),
    (20.6,10),
    (20.6,9.8),
    (20.6,10),
    (20.6,10.5),
    (20.6,10.7),
    (20.6,10.9),
    (20.6,11),
    (20.6,10.8),
    (18.7,10.4),
    (18.7,10.5),
    (18.7,10.7),
    (18.7,10.9),
    (18.7,10.6),
    (16,10),
    (16,10)
]
# %%

# wind should linearry transition to next average. Possibly we could have some normal sampling from that linear function as well
# gusts last max 20second and should only happen with maybe 1/8 or 1/10 of a chance

# the given inout to class should be seconds and for each second we should have corresponding wind state
# the wind state should either be normal wind or gust and we should transition to either gust (in seconds) or wind (in a minute or so)
# The input should be seconds and if it's over the given len of seconds of generated wind, we should start from the beginning
class wind_sampler:
    def __init__(self, time_scale_in_mins): 
        self.time_scale_in_mins = time_scale_in_mins
        self.time_scale_in_secs = time_scale_in_mins * 60
        self.initialize_wind(self.time_scale_in_secs)

    def initialize_wind(self, time_scale):

        # must find corresponding wind state to the second specific time data
        # multiply so that we match the time scale
        # TODO: what is the best way?
        # DO: count how many times the weather_data fits into the
        # time scale. use a loop that uses the int part of that
        # to append wind 
        # EXTRA: then we need to scale them so that there aren't "jumps"?
        # Gusts last about 3-20s
        
        extra_loops = time_scale // len(weather_data)

        for i in weather_data:
            print(i[1])

        return 0

    def sample_wind(self, current_second):
        return 0