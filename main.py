# ПОДКЛЮЧЕНИЕ БИБЛИОТЕК
import datetime
from math import pi, sin, cos, acos, asin
from pyorbital.orbital import Orbital
import matplotlib.pyplot as plt
import requests

# TLE, ПОЛУЧЕННОЕ С САЙТА

# КОЛИЧЕСТВО ДНЕЙ В МЕСЯЦЕ
month_list = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# РАДИУС ЗЕМЛИ
Re = 6378.137

# КООРДИНАТЫ ЛК
lon_0 = 55.93013
lat_0 = 37.51832
h = 0.198


# ФУНКЦИИ:

# ПРИБАВЛЕНИЕ 1 МИНУТЫ
def plus_minute(T):
    if T[4] != 59:
        T[4] += 1
    else:
        T[4] = 0
        if T[3] != 23:
            T[3] += 1
        else:
            T[3] = 0
            if T[2] != month_list[T[1]] - 1:
                T[2] += 1
            else:
                T[2] = 0
                if T[1] == 12:
                    T[1] = 0
                    T[0] += 1


# ПЕРЕВОД ИЗ МЕСТНОГО ВРЕМЕНИ
def minus_3_hours(T):
    if T[3] >= 3:
        T[3] -= 3
    else:
        T[3] = 24 + T[3] - 3
        if T[2] > 1:
            T[2] -= 1
        else:
            if T[1] > 1:
                T[1] -= 1
            else:
                T[0] -= 1
                T[1] = 12
            T[2] = month_list[T[1] - 1]
    return T

# СТРОКУ, ВВОДИМУЮ С КЛАВИАТУРЫ, ПРЕВРАЩАЕМ В СПИСОК ЦЕЛОЧИСЛЕННЫХ ПЕРЕМЕННЫХ:

def string_to_list(str):
    return [int(str[0:4]), int(str[5:7]), int(str[8:10]), int(str[11:13]), int(str[14:16])]


# СПИСОК ВЫШЕ ПРЕВРАЩАЕМ В ВРЕМЯ ФОРМАТА DATETIME
def to_datetime(list):
    return datetime.datetime(list[0], list[1], list[2], list[3], list[4])


# ПЕРЕВОД УГЛА В РАДИАНЫ
def to_rad(phi):
    return phi / 180 * pi


# ПЕРЕВОД УГЛА В ГРАДУСЫ
def to_deg(phi):
    return phi / pi * 180


# ПЕРЕХОД ИЗ ПОЛЯРНОЙ СК В ДЕКАРТОВУ
def from_polar(r, lon, lat):
    x = r * cos(lon) * cos(lat)
    y = r * sin(lon) * cos(lat)
    z = r * sin(lat)
    return x, y, z


# ДЛИНА ВЕКТОРА
def module(x, y, z):
    return (x ** 2 + y ** 2 + z ** 2) ** 0.5


# СКАЛЯРНОЕ ПРОИЗВЕДЕНИЕ ВЕКТОРОВ
def scale(x, y, z, a, b, c):
    return x * a + y * b + z * c


# НАПРАВЛЕНИЕ НА СЕВЕР
def north(x, y, z):
    D = -x ** 2 - y ** 2 - z ** 2
    x_north = 0
    y_north = 0
    z_north = -D / z0
    return x_north, y_north, z_north


# КООРДИНАТЫ X, Y, Z НАШЕГО МЕСТОПОЛОЖЕНИЯ
lon_0 = to_rad(lon_0)
lat_0 = to_rad(lat_0)
x0, y0, z0 = from_polar(Re + h, lon_0, lat_0)

# КООРДИНАТЫ ВЕКТОРА НАПРАВЛЕНИЯ НА СЕВЕР В НАШЕЙ ПЛОСКОСТИ - i
xn, yn, zn = north(x0, y0, z0)
xi, yi, zi = xn - x0, yn - y0, zn - z0

# КООРДИНАТЫ ВЕКТОРА НАПРАВЛЕНИЯ НА ВОСТОК В НАШЕЙ ПЛОСКОСТИ - j
xj, yj, zj = yi * z0 - zi * y0, -(xi * z0 - x0 * zi), xi * y0 - yi * x0

# ВВОД ВРЕМЕНИ НАЧАЛА И КОНЦА ПРОГНОЗА
print("Enter time start: YYYY MM DD HH mm   (f. e. '2021 03 01 08 05' - The 1st of March 2021, 8:05 am)")
T_start = input()
print("Enter time end:   YYYY MM DD HH mm   (f. e. '2021 05 05 22 50' - The 5th of May   2021, 10:50 pm)")
T_end = input()

T_lstart = string_to_list(T_start)
T_lend = string_to_list(T_end)

T_out_start = string_to_list(T_start)
T_out_end = string_to_list(T_end)

T1 = to_datetime(T_lstart)
T2 = to_datetime(T_lend)

print("Time start: ", T1)
print("Time end:   ", T2)

# ПЕРЕВОД ВРЕМЕНИ НАЧАЛА И КОНЦА В UTC
T_lstart = minus_3_hours(T_lstart)
T_lend = minus_3_hours(T_lend)

# СПИСКИ, В КОТОРЫЕ БУДУТ ЗАПИСЫВАТЬСЯ НАШИ ДАННЫЕ
Lx = []
Ly = []
Lz = []
elevation = []
azimut = []

# СОЗДАНИЕ ОБЪЕКТА ORBITAL ДЛЯ НАШЕГО СПУТНИКА (С ЕГО TLE)
orb = Orbital("NOAA 19")


# ОСНОВНАЯ ФУНКЦИЯ - НАХОЖДЕНИЕ КООРДИНАТ СПУТНИКА, ЭЛЕВАЦИИ И АЗИМУТА
def get_prognose(T, time):
    # ПОЛУЧЕНИЕ ШИРОТЫ, ДОЛГОТЫ И РАССТОЯНИЯ ДО ЗЕМЛИ СПУТНИКА
    lon, lat, alt = orb.get_lonlatalt(T)

    # ПЕРЕВОД В РАДИАНЫ + ПРИБАВЛЕНИЕ РАДИУСА ЗЕМЛИ
    a = to_rad(lon)
    b = to_rad(lat)
    r = alt + Re

    # НАХОЖДЕНИЕ КООРДИНАТ СПУТНИКА
    x, y, z = from_polar(r, a, b)
    Lx.append(x)
    Ly.append(y)
    Lz.append(z)


    # КАСАТЕЛЬНАЯ ПЛОСКОСТЬ К ПОВЕРХНОСТИ ЗЕМЛИ
    A = x0
    B = y0
    C = z0
    D = -x0 ** 2 - y0 ** 2 - z0 ** 2

    # РАССТОЯНИЕ ОТ ТОЧКИ ДО ПЛОСКОСТИ
    Po = (A * x + B * y + C * z + D) / ((A ** 2 + B ** 2 + C ** 2) ** 0.5)

    # ДЛИНА ВЕКТОРА ОТ НАШЕГО ПОЛОЖЕНИЯ ДО СПУТНИКА
    r1 = module(x - x0, y - y0, z - z0)

    # ЭЛЕВАЦИЯ
    Elevation = asin(scale(x-x0, y-y0, z-z0, x0, y0, z0)/r1/module(x0, y0, z0))
    Elevation = to_deg(Elevation)

    # ПРОЕКЦИЯ ТОЧКИ СПУТНИКА НА КАСАТЕЛЬНУЮ ПЛОСКОСТЬ
    t = (x0 ** 2 - x0 * x + y0 ** 2 - y0 * y + z0 ** 2 - z0 * z) / (x0 ** 2 + y0 ** 2 + z0 ** 2)
    xp = x + t*x0
    yp = y + t*y0
    zp = z + t*z0

    # ПЕРЕХОД ОТ ТОЧКИ К ВЕКТОРУ, СОЕДИНЯЮЩЕМУ НАШУ ТОЧКУ С ТОЧКОЙ ПРОЕКЦИИ
    xp = xp - x0
    yp = yp - y0
    zp = zp - z0

    # УГОЛ МЕЖДУ НАПРАВЛЕНИЕМ НА СЕВЕР И ВЕКТОРОМ ПРОЕКЦИИ
    N = acos(scale(xp, yp, zp, xi, yi, zi) / module(xp, yp, zp) / module(xi, yi, zi))
    N = to_deg(N)

    # УГОЛ МЕЖДУ НАПРАВЛЕНИЕМ НА ВОСТОК И ВЕКТОРОМ ПРОЕКЦИИ
    E = acos(scale(xp, yp, zp, xj, yj, zj) / module(xp, yp, zp) / module(xj, yj, zj))
    E = to_deg(E)

    # ПОПРАВКА С УЧЕТОМ НАПРАВЛЕНИЯ НА ВОСТОК до 360 градусов
    if E > 90:
        N = 360 - N

    # АЗИМУТ
    Azimut = N

    if Elevation>10:
        print(to_datetime(time), " 0El:  ", Elevation, "  Az: ", Azimut)
    elevation.append(Elevation)
    azimut.append(Azimut)


# ЦИКЛ ВЫПОЛНЕНИЯ ФУНКЦИИ
T_out = T_out_start
T_work = T_lstart
while T_work != T_lend:
    T = to_datetime(T_work)
    get_prognose(T, T_out)
    plus_minute(T_work)
    plus_minute(T_out)

# ГРАФИК ДВИЖЕНИЯ СПУТНИКА
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(Lx, Ly, Lz)
fig.set_size_inches(7, 7)
plt.show()

# РАЗБИЕНИЕ НА ПРОХОЖДЕНИЯ

elevation_list = []
azimut_list = []


single_elevation = []
single_azimut = []

for i in range(len(elevation)):
    if elevation[i] > 0:
        single_elevation.append(elevation[i])
        single_azimut.append(to_rad(azimut[i]))
    else:
        if single_elevation:
            elevation_list.append(single_elevation)
            azimut_list.append(single_azimut)
        single_elevation = []
        single_azimut = []


# ПОЛЯРНЫЙ ГРАФИК

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
ax.set_theta_zero_location('N')
ax.set_rlim(bottom=90, top=0)
ax.set_theta_direction(-1)
for phi, theta in zip(azimut_list, elevation_list):
    ax.plot(phi, theta)
fig.set_size_inches(7, 7)
plt.show()
