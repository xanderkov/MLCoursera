def write_answer_to_file(answer, filename):
    with open(filename, 'w') as f_out:
        f_out.write(str(round(answer, 3)))


# **1. Загрузите данные из файла *advertising.csv* в объект pandas DataFrame. [Источник данных](http://www-bcf.usc.edu/~gareth/ISL/data.html).**

# In[2]:

import pandas as pd
adver_data = pd.read_csv('advertising.csv')


# **Посмотрите на первые 5 записей и на статистику признаков в этом наборе данных.**

# In[3]:

adver_data.head(n=5)


# In[4]:

adver_data.describe()


# **Создайте массивы NumPy *X* из столбцов TV, Radio и Newspaper и *y* - из столбца Sales. Используйте атрибут *values* объекта pandas DataFrame.**

# In[5]:

X = adver_data[["TV","Radio", "Newspaper"]].values
Y = adver_data[["Sales"]].values


# **Отмасштабируйте столбцы матрицы *X*, вычтя из каждого значения среднее по соответствующему столбцу и поделив результат на стандартное отклонение.**

# In[6]:

import numpy as np
means, stds = np.mean(X, axis=0), np.std(X, axis=0)


# In[7]:

X = (X - means)/stds


# **Добавьте к матрице *X* столбец из единиц, используя методы *hstack*, *ones* и *reshape* библиотеки NumPy. Вектор из единиц нужен для того, чтобы не обрабатывать отдельно коэффициент $w_0$ линейной регрессии.**

# In[8]:

import numpy as np
X = np.hstack([X, np.ones((X.shape[0],1))])


# **2. Реализуйте функцию *mserror* - среднеквадратичную ошибку прогноза. Она принимает два аргумента - объекты Series *y* (значения целевого признака) и *y\_pred* (предсказанные значения).**

# In[9]:

def mserror(y, y_pred):
    return round((sum((y - y_pred)**2)[0])/float(y.shape[0]), 3)


# **Какова среднеквадратичная ошибка прогноза значений Sales, если всегда предсказывать медианное значение Sales по исходной выборке? Запишите ответ в файл '1.txt'.**

# In[10]:

eye = np.array([np.median(Y)]*Y.shape[0]).reshape((Y.shape[0], 1))
answer1 = mserror(Y, eye)
print(answer1)
write_answer_to_file(answer1, '1.txt')


# **3. Реализуйте функцию *normal_equation*, которая по заданным матрицам (массивам NumPy) *X* и *y* вычисляет вектор весов $w$ согласно нормальному уравнению линейной регрессии.**

# In[11]:

def normal_equation(X, y):
    return np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)


# In[12]:

norm_eq_weights = normal_equation(X, Y)
print(norm_eq_weights)


# **Какие продажи предсказываются линейной моделью с весами, найденными с помощью нормального уравнения, в случае средних инвестиций в рекламу по ТВ, радио и в газетах? (то есть при нулевых значениях масштабированных признаков TV, Radio и Newspaper). Запишите ответ в файл '2.txt'.**

# In[13]:

answer2 = np.dot(np.mean(X, axis=0), norm_eq_weights)[0]
print(answer2)
write_answer_to_file(answer2, '2.txt')


# **4. Напишите функцию *linear_prediction*, которая принимает на вход матрицу *X* и вектор весов линейной модели *w*, а возвращает вектор прогнозов в виде линейной комбинации столбцов матрицы *X* с весами *w*.**

# In[14]:

def linear_prediction(X, w):
    return np.dot(X, w)


# **Какова среднеквадратичная ошибка прогноза значений Sales в виде линейной модели с весами, найденными с помощью нормального уравнения? Запишите ответ в файл '3.txt'.**

# In[15]:

answer3 = mserror(Y, linear_prediction(X, norm_eq_weights))
print(answer3)
write_answer_to_file(answer3, '3.txt')


# **5. Напишите функцию *stochastic_gradient_step*, реализующую шаг стохастического градиентного спуска для линейной регрессии. Функция должна принимать матрицу *X*, вектора *y* и *w*, число *train_ind* - индекс объекта обучающей выборки (строки матрицы *X*), по которому считается изменение весов, а также число *$\eta$* (eta) - шаг градиентного спуска (по умолчанию *eta*=0.01). Результатом будет вектор обновленных весов.**

# In[16]:

def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
    x_k = X[train_ind, :]
    y_k = y[train_ind]
    y_pred = np.dot(x_k, w)
    l = X.shape[0]
    return w + (2*eta/l)*(y_k - y_pred)*x_k


# **6. Напишите функцию *stochastic_gradient_descent*, реализующую стохастический градиентный спуск для линейной регрессии. Функция принимает на вход следующие аргументы:**
# - X - матрица, соответствующая обучающей выборке
# - y - вектор значений целевого признака
# - w_init - вектор начальных весов модели
# - eta - шаг градиентного спуска (по умолчанию 0.01)
# - max_iter - максимальное число итераций градиентного спуска (по умолчанию 10000)
# - max_weight_dist - минимальное евклидово расстояние между векторами весов на соседних итерациях градиентного спуска,
# при котором алгоритм прекращает работу (по умолчанию 1e-8)
# - seed - число, используемое для воспроизводимости сгенерированных псевдослучайных чисел (по умолчанию 42)
# - verbose - флаг печати информации (например, для отладки, по умолчанию False)
# 
# **На каждой итерации в вектор (список) должно записываться текущее значение среднеквадратичной ошибки. Функция должна возвращать вектор весов $w$, а также вектор (список) ошибок.**

# In[40]:

def stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4,
                                min_weight_dist=1e-8, seed=42, verbose=False):
    # Инициализируем расстояние между векторами весов на соседних
    # итерациях большим числом. 
    weight_dist = np.inf
    # Инициализируем вектор весов
    w = w_init
    # Сюда будем записывать ошибки на каждой итерации
    errors = []
    # Счетчик итераций
    iter_num = 0
    # Будем порождать псевдослучайные числа 
    # (номер объекта, который будет менять веса), а для воспроизводимости
    # этой последовательности псевдослучайных чисел используем seed.
    np.random.seed(seed)
        
    # Основной цикл
    while weight_dist > min_weight_dist and iter_num < max_iter:
        # порождаем псевдослучайный 
        # индекс объекта обучающей выборки
        random_ind = np.random.randint(X.shape[0])
        
        # Ваш код здесь
        old_w = w
        w = stochastic_gradient_step(X, y, w, random_ind, eta=eta)
        weight_dist = np.linalg.norm(w - old_w)
        errors.append(mserror(y, np.dot(X, w)))
        iter_num += 1
        
        if iter_num % 10000 == 0 and verbose:
            print "Iteration: ", iter_num
        
    return w, errors


#  **Запустите $10^5$ итераций стохастического градиентного спуска. Укажите вектор начальных весов *w_init*, состоящий из нулей. Оставьте параметры  *eta* и *seed* равными их значениям по умолчанию (*eta*=0.01, *seed*=42 - это важно для проверки ответов).**

# In[41]:

get_ipython().run_cell_magic(u'time', u'', u'stoch_grad_desc_weights, stoch_errors_by_iter = stochastic_gradient_descent(X, Y, np.ones((X.shape[1])), eta=1e-2, max_iter=10**5, verbose = True)')


# In[42]:

get_ipython().magic(u'pylab inline')
plot(range(len(stoch_errors_by_iter)), stoch_errors_by_iter)
xlabel('Iteration number')
ylabel('MSE')


# **Посмотрим на вектор весов, к которому сошелся метод.**

# In[43]:

stoch_grad_desc_weights


# **Посмотрим на среднеквадратичную ошибку на последней итерации.**

# In[44]:

stoch_errors_by_iter[-1]


# **Какова среднеквадратичная ошибка прогноза значений Sales в виде линейной модели с весами, найденными с помощью градиентного спуска? Запишите ответ в файл '4.txt'.**

# In[61]:

print sum((Y - np.dot(X, stoch_grad_desc_weights).reshape((Y.shape[0], 1)))**2)/float(Y.shape[0])
answer4 = round(np.mean((Y - np.dot(X, stoch_grad_desc_weights).reshape((Y.shape[0], 1)))**2), 3)
print(answer4)
write_answer_to_file(answer4, '4.txt')