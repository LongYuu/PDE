from flask import Flask, jsonify, request
from flask_cors import CORS
import sklearn
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from nn import NeuralNetwork
import random
import pyEDM
from sklearn.preprocessing import StandardScaler
from PyIF import te_compute as te
import nitime
import nitime.analysis as nta
import nitime.timeseries as ts
import nitime.utils as tsu


# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CROS
CORS(app, resources={r'/*': {'origins': '*'}})

# sanity check route
@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify('pong!')

MODEL_INFO = {
    'fileName': '',
    'layers': '',
    'arr': [],
    'neurons': [],
    'activation': '',
    'loss': [],
    'history': [],
    'variables': '',
}
@app.route('/model', methods=['GET', 'POST'])
def build_model():
    global MODEL
    if MODEL_INFO['fileName'] == '':
        response_object = {'status': 'file is not loaded'}
    elif MODEL_INFO['loss'] == []:
        response_object = {'status': 'training the model'}
    else:
        response_object = {'status': 'success'}

    if request.method == 'POST':
        post_data = request.get_json()
        MODEL_INFO['fileName'] = post_data.get('fileName')
        path = './data/' + MODEL_INFO['fileName']
        MODEL_INFO['layers'] = post_data.get('layers')
        MODEL_INFO['arr'] = []
        for i in range(MODEL_INFO['layers']):
            MODEL_INFO['arr'].append('L' + str(i + 1))
        MODEL_INFO['neurons'] = post_data.get('neurons')
        MODEL_INFO['activation'] = post_data.get('activation')
        # load dataset from csv file
        df = pd.read_csv(path)
        # shape of dataset
        (row, col) = df.shape
        # split dataset to traning(70%) and validation(30%) then training
        u = df['u'].values
        t = df['t'].values
        if col == 3:
            MODEL_INFO['variables'] = 1
            x = df['x'].values

            N_train = row // 10 * 7
            idx = np.random.choice(x.shape[0], N_train, replace=False)
            x_train = x[idx].reshape(N_train,1)
            t_train = t[idx].reshape(N_train,1)
            u_train = u[idx].reshape(N_train,1)
            N_val = row - N_train
            idx2 = np.random.choice(x.shape[0], N_val, replace=False)
            x_val = x[idx2].reshape(N_val,1)
            t_val = t[idx2].reshape(N_val,1)

            layers = [2]
            for each in MODEL_INFO['neurons']:
                layers.append(each)
            layers.append(1)

            model = NeuralNetwork(x_train, t_train, u_train, layers, MODEL_INFO['activation'])
            MODEL = model
            MODEL_INFO['loss'].append(model.train(1000, MODEL_INFO['variables']))
            MODEL_INFO['history'].append((MODEL_INFO['layers'], MODEL_INFO['neurons'], MODEL_INFO['loss'][-1][-1]))
        elif col == 4:
            MODEL_INFO['variables'] = 2
            x = df['x'].values
            y = df['y'].values

            N_train = row // 10 * 7
            idx = np.random.choice(x.shape[0], N_train, replace=False)
            x_train = x[idx].reshape(N_train,1)
            y_train = y[idx].reshape(N_train,1)
            t_train = t[idx].reshape(N_train,1)
            u_train = u[idx].reshape(N_train,1)
            N_val = row - N_train
            idx2 = np.random.choice(x.shape[0], N_val, replace=False)
            x_val = x[idx2].reshape(N_val,1)
            y_val = y[idx2].reshape(N_val,1)
            t_val = t[idx2].reshape(N_val,1)

            layers = [3]
            for each in MODEL_INFO['neurons']:
                layers.append(each)
            layers.append(1)

            model = NeuralNetwork(x_train, y_train, t_train, u_train, layers, MODEL_INFO['activation'])
            MODEL = model
            MODEL_INFO['loss'].append(model.train(50000, MODEL_INFO['variables']))
            MODEL_INFO['history'].append((MODEL_INFO['layers'], MODEL_INFO['neurons'], MODEL_INFO['loss'][-1][-1]))
        elif col == 5:
            MODEL_INFO['variables'] = 3
            x = df['x'].values
            y = df['y'].values
            z = df['z'].values

            N_train = row // 10 * 7
            idx = np.random.choice(x.shape[0], N_train, replace=False)
            x_train = x[idx].reshape(N_train,1)
            y_train = y[idx].reshape(N_train,1)
            z_train = z[idx].reshape(N_train,1)
            t_train = t[idx].reshape(N_train,1)
            u_train = u[idx].reshape(N_train,1)
            N_val = row - N_train
            idx2 = np.random.choice(x.shape[0], N_val, replace=False)
            x_val = x[idx2].reshape(N_val,1)
            y_val = y[idx2].reshape(N_val,1)
            z_val = z[idx2].reshape(N_val,1)
            t_val = t[idx2].reshape(N_val,1)

            layers = [4]
            for each in MODEL_INFO['neurons']:
                layers.append(int(each))
            layers.append(1)

            model = NeuralNetwork(x_train, y_train, z_train, t_train, u_train, layers, MODEL_INFO['activation'])
            MODEL = model
            MODEL_INFO['loss'].append(model.train(1000, MODEL_INFO['variables']))
            MODEL_INFO['history'].append((MODEL_INFO['layers'], MODEL_INFO['neurons'], MODEL_INFO['loss'][-1][-1]))
        else:
            response_object = {'status': 'please format the input file with index["x", "y", "z", "t", "u"]'}
            return jsonify(response_object)
    else:
        response_object['model_info'] = MODEL_INFO

    return jsonify(response_object)

PREDICTION = {
    'predMin': 10000,
    'predMax': 0,
    'triData': [],
    'time': '',
}

def get_model_pred(time):
    # model = keras.models.load_model('./savedmodel/{}/model_{:0>2d}_{:0>2d}.h5'.format(model_type, hidden_layers, layer_size))
    global MODEL
    model = MODEL

    if MODEL_INFO['variables'] == 3:
        # calculate the prediction for 3d plotting
        single_point = np.zeros((1, 4))
        values = []

        index = 0
        minV = 10000
        maxV = 0
        for i in range(10):
            x = i * (1. / (10.-1.))
            for j in range(10):
                y = j * (1. / (10.-1.))
                for k in range(10):
                    z = k * (1. / (10.-1.))

                    single_point[0, 0] = x
                    single_point[0, 1] = y
                    single_point[0, 2] = z
                    single_point[0, 3] = float(time)
                    tmp = model.predict(single_point, 3)[0]
                    values.append([x, y, z, tmp.item()])
                    if values[-1][3] < minV:
                        minV = values[-1][3]
                    if values[-1][3] > maxV:
                        maxV = values[-1][3]
                    index = index + 1

    elif MODEL_INFO['variables'] == 2:
        # calculate the prediction for 3d plotting
        single_point = np.zeros((1, 3))
        values = []

        index = 0
        minV = 10000
        maxV = 0
        for i in range(10):
            x = i * (1. / (10.-1.))
            for j in range(10):
                y = j * (1. / (10.-1.))
                for k in range(10):
                    z = k * (1. / (10.-1.))

                    single_point[0, 0] = x
                    single_point[0, 1] = y
                    # single_point[0, 2] = z
                    single_point[0, 2] = float(time)
                    tmp = model.predict(single_point, 2)[0]
                    values.append([x, y, z, tmp.item()])
                    if values[-1][3] < minV:
                        minV = values[-1][3]
                    if values[-1][3] > maxV:
                        maxV = values[-1][3]
                    index = index + 1
        
    return [minV, maxV, values]

@app.route('/prediction', methods=['GET', 'POST'])
def model_pred():
    if MODEL_INFO['fileName'] == '':
        response_object = {'status': 'file is not loaded'}
    elif MODEL_INFO['loss'] == []:
        response_object = {'status': 'training the model'}
    else:
        response_object = {'status': 'success'}
        if request.method == 'POST':
            post_data = request.get_json()
            PREDICTION['time'] = post_data.get('time')
            tmp = get_model_pred(PREDICTION['time'])
            PREDICTION['predMin'] = tmp[0]
            PREDICTION['predMax'] = tmp[1]
            PREDICTION['triData'] = tmp[2]
        else:
            response_object['prediction'] = PREDICTION

    return jsonify(response_object)

# def generate_1d_ts(x, library):
#     global MODEL
#     model = MODEL

#     t_window = [0., 1.]
#     num = library
#     sample_points = np.zeros((num, 4))
#     for i in range(num):
#         sample_points[i, 0] = x
#         sample_points[i, 1] = t_window[0] + i * (t_window[1] - t_window[0]) / num

#     input_tensor = tf.Variable(sample_points, dtype=tf.float32)
#     tmp = model.predict(input_tensor, 3) # u, u_t, u_x, u_xx, u_xxx
#     return tmp

def generate_2d_ts(x, y, library):
    global MODEL
    model = MODEL

    t_window = [0., 1.]
    num = int(library)
    sample_points = np.zeros((num, 3))
    for i in range(num):
        sample_points[i, 0] = x
        sample_points[i, 1] = y
        sample_points[i, 2] = t_window[0] + i * (t_window[1] - t_window[0]) / num

    tmp = model.predict(sample_points, 2) # u, u_t, u_x, u_y, u_xx, u_xy, u_yy, u_xxx, u_xxy, u_xyy, u_yyy
    return tmp

def generate_3d_ts(x, y, z, library):
    global MODEL
    model = MODEL

    t_window = [0., 1.]
    num = int(library)
    sample_points = np.zeros((num, 4))
    for i in range(num):
        sample_points[i, 0] = float(x)
        sample_points[i, 1] = float(y)
        sample_points[i, 2] = float(z)
        sample_points[i, 3] = t_window[0] + i * (t_window[1] - t_window[0]) / num

    tmp = model.predict(sample_points, 3) # u, u_t, u_x, u_y, u_z, u_xx, u_xy, u_xz, u_yy, u_yz, u_zz, u_xxx, u_xxy, u_xxz, u_xyy, u_xyz, u_xzz, u_yyy, u_yyz, u_yzz, u_zzz
    return tmp

def calculate_ccm(df, objective):
    result = []
    Candidates = df
    data = Candidates.values
    scaler = StandardScaler(with_mean=True, with_std=True)
    data = scaler.fit_transform(data)
    time_index = [int(i) for i in range(df.shape[0])]
    edm_data = pd.DataFrame(data, index=time_index, columns=Candidates.columns)
    edm_data.insert(loc=0, column='Time', value=range(len(edm_data)))
    
    bestEDim = pd.DataFrame(np.zeros((len(edm_data.columns[1:]), 2)), index=edm_data.columns[1:], columns=['E', 'MAE'])
    for i, sp in enumerate(edm_data.columns[1:]):
        MAEs = np.zeros(len(range(2,25)))
        for E in range(2, 25):
            library_string = "1 {}".format(len(edm_data) - E)
            preds = pyEDM.Simplex(dataFrame=edm_data, columns=sp, target=sp, \
                        E=E, Tp=1, lib=library_string, pred=library_string)
            MAEs[E-2] = np.nanmean(np.abs((preds['Predictions'] - preds['Observations']).values))

        best_E = np.argmin(MAEs) + 2
        bestEDim.loc[sp, 'E'] = best_E
        bestEDim.loc[sp, 'MAE'] = MAEs[best_E - 2]

    THETAs = np.arange(0.1, 3.1, 0.1)
    # local
    result.append(THETAs.tolist())
    for i, sp in enumerate(edm_data.columns[1:]):
        E = int(bestEDim.loc[sp, 'E'])
        library_string =  "1 {}".format(len(edm_data) - E)
        MAEs = np.zeros(len(THETAs))
        for t, theta in enumerate(THETAs):
            temp = pyEDM.SMap(dataFrame=edm_data, columns=sp, target=sp, \
                        E=E, Tp=1, lib=library_string, pred=library_string, theta=theta)
            mae = np.abs((temp['predictions']['Predictions'] - temp['predictions']['Observations']).values[1:-1]).mean()
            MAEs[t] = mae
        if sp == 'u_t':
            # mae
            result.append(MAEs.tolist())

    import itertools
    sp1 = objective 
    sp2 = "u_t"
    edim1 = int(bestEDim.loc[sp1, 'E'])
    edim2 = int(bestEDim.loc[sp2, 'E']) 
    # print('From:', sp1, ' To:', sp2, ' E =', edim1)
    results1 = pyEDM.CCM(dataFrame = edm_data[['Time', sp1, sp2]], \
        E = edim1, Tp = 0, columns = sp1, target = sp2, \
        libSizes = f"{edim1} 100 10", sample = 100, verbose = True,  showPlot = False)
    # libsize
    result.append(results1.values[:, 0].tolist())
    # cor
    result.append(results1.values[:, 1].tolist())
    # print('From:', sp2, ' To:', sp1, ' E =', edim2)
    # results2 = pyEDM.CCM(dataFrame = edm_data[['Time', sp1, sp2]], \
    #     E = edim2, Tp = 0, columns = sp2, target = sp1, \
    #     libSizes = f"{edim2} 100 10", sample = 100, verbose = True,  showPlot = False) 

    return result

def calculate_te(df, objective):
    result = []

    col = df.shape[0]
    lib = list(range(col // 10, col+1, col // 10))
    result.append(lib)
    result.append([])
    for i in lib:
        dt = df['u_t'].values[:i]
        value = df[objective].values[:i]
        result[1].append(te.te_compute(value, dt, k=1, embedding=1, safetyCheck=True, GPU=False))

    return result

def calculate_gc(df, objective):
    result = []

    col = df.shape[0]
    lib = list(range(col // 10, col+1, col // 10))
    result.append(lib)
    result.append([])
    for i in lib:
        dt = df['u_t'].values[:i].reshape(1, i)
        value = df[objective].values[:i].reshape(1, i)
        data = np.concatenate((dt, value), 0)
        pdata = tsu.percent_change(data)
        time_series = ts.TimeSeries(pdata, sampling_interval=0.01)
        G = nta.GrangerAnalyzer(time_series, order=1)
        result[1].append( np.mean(G.causality_yx[:, :], -1)[0][1] )
    
    # acf
    nlags = 20
    x = df[objective].values.reshape(col, 1)
    demo = 0
    mole_list = []
    avg = sum(x)/len(x)
    for i in range(len(x)):
        demo+=(x[i]-avg)**2
    demo = demo/len(x)
    for i in range(len(x)):
        demo+=(x[i]-avg)**2
    demo = demo/len(x)
    for i in range(1, nlags+1):
        mole = 0
        list1 = x[i:]
        list2 = x[:-i]
        avg1 = sum(list1)/len(list1)
        avg2 = sum(list2)/len(list2)
        for i in range(len(list1)):
            mole+=(list1[i]-avg1)*(list2[i]-avg2)
        mole = mole/len(list1)
        mole_list.append(mole)
    result_acf = [(m/demo).tolist()[0] for m in mole_list]
    result_acf.insert(0, 1)
    result.append(result_acf)
    
    return result

CAUSALITY = {
    'library': '',
    'x': '1.0',
    'y': '1.0',
    'z': '1.0',
    'candidates': [],
    'objective': '',
    'triData': [],
    'triMax': '',
    'triMin': '',
    'mae': [],
    'minmae': '',
    'maxmae': '',
    'local': [],
    'cor': [],
    'libsize': [],
    'type': '',
    'u': [],
    'dt': [],
    'dx': [],
    'dy': [],
    'dz': [],
    'dxx': [],
    'dxy': [],
    'dxz': [],
    'dyy': [],
    'dyz': [],
    'dzz': [],
    'ts': [],
    'acf': [],
} 
@app.route('/causality_lib', methods=['GET', 'POST'])
def causality_lib():
    response_object = {'status': 'success'}
    if request.method == 'POST':
        post_data = request.get_json()
        CAUSALITY['library'] = post_data.get('library')
        CAUSALITY['ts'] = np.linspace(0.0, 1.0, num=int(CAUSALITY['library'])-1).tolist()
        tmp = generate_3d_ts(CAUSALITY['x'], CAUSALITY['y'], CAUSALITY['z'], CAUSALITY['library'])
        CAUSALITY['u'] = tmp[0].flatten().tolist()
        CAUSALITY['dt'] = tmp[1].flatten().tolist()
        CAUSALITY['dx'] = tmp[2].flatten().tolist()
        CAUSALITY['dy'] = tmp[3].flatten().tolist()
        CAUSALITY['dz'] = tmp[4].flatten().tolist()
        CAUSALITY['dxx'] = tmp[5].flatten().tolist()
        CAUSALITY['dxy'] = tmp[6].flatten().tolist()
        CAUSALITY['dxz'] = tmp[7].flatten().tolist()
        CAUSALITY['dyy'] = tmp[8].flatten().tolist()
        CAUSALITY['dyz'] = tmp[9].flatten().tolist()
        CAUSALITY['dzz'] = tmp[10].flatten().tolist()
        # # u, u_t, u_x, u_y, u_xx, u_xy, u_yy, u_xxx, u_xxy, u_xyy, u_yyy
        # tmp = generate_2d_ts(CAUSALITY['x'], CAUSALITY['y'], CAUSALITY['library'])
        # CAUSALITY['u'] = tmp[0].flatten().tolist()
        # CAUSALITY['dt'] = tmp[1].flatten().tolist()
        # CAUSALITY['dx'] = tmp[2].flatten().tolist()
        # CAUSALITY['dy'] = tmp[3].flatten().tolist()
        # CAUSALITY['dxx'] = tmp[4].flatten().tolist()
        # CAUSALITY['dxy'] = tmp[5].flatten().tolist()
        # CAUSALITY['dyy'] = tmp[6].flatten().tolist()
        # CAUSALITY['dxxx'] = tmp[7].flatten().tolist()
        # CAUSALITY['dxxy'] = tmp[8].flatten().tolist()
        # CAUSALITY['dxyy'] = tmp[9].flatten().tolist()
        # CAUSALITY['dyyy'] = tmp[10].flatten().tolist()
    else:
        response_object['causality_lib'] = CAUSALITY

    return jsonify(response_object)

@app.route('/causality_candidate', methods=['GET', 'POST'])
def causality_candidate():
    response_object = {'status': 'success'}
    if request.method == 'POST':
        post_data = request.get_json()
        CAUSALITY['candidates'] = post_data.get('candidates')
    else:
        response_object['causality_candidates'] = CAUSALITY

    return jsonify(response_object)

@app.route('/causality', methods=['GET', 'POST'])
def causality():
    if CAUSALITY['library'] == '':
        response_object = {'status': 'Need to generate time series data from neural network model'}
    else:
        response_object = {'status': 'success'}
        if request.method == 'POST':
            post_data = request.get_json()
            CAUSALITY['x'] = post_data.get('x')
            CAUSALITY['y'] = post_data.get('y')
            CAUSALITY['z'] = post_data.get('z')
            CAUSALITY['type'] = post_data.get('type')
            CAUSALITY['objective'] = post_data.get('objective')
            tmp = generate_3d_ts(CAUSALITY['x'], CAUSALITY['y'], CAUSALITY['z'], CAUSALITY['library'])
            # tmp = generate_2d_ts(CAUSALITY['x'], CAUSALITY['y'], CAUSALITY['library'])
            for i in range(len(tmp)):
                if i == 0:
                    values = np.array(tmp[0])
                else:
                    values = np.concatenate((values, tmp[i]), 1)
            df = pd.DataFrame(values, columns=['u', 'u_t', 'u_x', 'u_y', 'u_z', 'u_xx', 'u_xy', 'u_xz', 'u_yy', 'u_yz', 'u_zz', 'u_xxx', 'u_xxy', 'u_xxz', 'u_xyy', 'u_xyz', 'u_xzz', 'u_yyy', 'u_yyz', 'u_yzz', 'u_zzz'])
            # df = pd.DataFrame(values, columns=['u', 'u_t', 'u_x', 'u_y', 'u_xx', 'u_xy', 'u_yy', 'u_xxx', 'u_xxy', 'u_xyy', 'u_yyy'])
            # Get the data for plotting
            if CAUSALITY['type'] == 'ccm':
                # values = np.zeros((int(CAUSALITY['library']), 1), dtype = np.float32)
                result = calculate_ccm(df, CAUSALITY['objective'])
                CAUSALITY['local'] = result[0]
                CAUSALITY['mae'] = result[1]
                CAUSALITY['minmae'] = min(result[1])
                CAUSALITY['maxmae'] = max(result[1])
                CAUSALITY['libsize'] = result[2]
                CAUSALITY['cor'] = result[3]
            elif CAUSALITY['type'] == 'te':
                result = calculate_te(df, CAUSALITY['objective'])
                CAUSALITY['libsize'] = result[0]
                CAUSALITY['cor'] = result[1]
            elif CAUSALITY['type'] == 'gc':
                result = calculate_gc(df, CAUSALITY['objective'])
                CAUSALITY['libsize'] = result[0]
                CAUSALITY['cor'] = result[1]
                CAUSALITY['acf'] = result[2]

        else:
            response_object['causality_info'] = CAUSALITY

    return jsonify(response_object)

@app.route('/causality_volume', methods=['GET', 'POST'])
def causality_volume():
    if CAUSALITY['library'] == '':
        response_object = {'status': 'Need to generate time series data from neural network model'}
    elif CAUSALITY['type'] == '':
        response_object = {'status': 'Need to select a causality detection method'}
    else:
        response_object = {'status': 'success'}
        if request.method == 'POST':
            post_data = request.get_json()
            CAUSALITY['type'] = post_data.get('type')
            CAUSALITY['objective'] = post_data.get('objective')
            # calculate the prediction for 3d plotting
            triValues = []
            minV = 10000
            maxV = 0
            for i in range(10):
                x = i * (1. / (10.-1.))
                for j in range(10):
                    y = j * (1. / (10.-1.))
                    for k in range(10):
                        z = k * (1. / (10.-1.))
                        tmp = generate_3d_ts(x, y, z, CAUSALITY['library'])
                        # tmp = generate_2d_ts(x, y, CAUSALITY['library'])
                        for i in range(len(tmp)):
                            if i == 0:
                                values = np.array(tmp[0])
                            else:
                                values = np.concatenate((values, tmp[i]), 1)
                        df = pd.DataFrame(values, columns=['u', 'u_t', 'u_x', 'u_y', 'u_z', 'u_xx', 'u_xy', 'u_xz', 'u_yy', 'u_yz', 'u_zz', 'u_xxx', 'u_xxy', 'u_xxz', 'u_xyy', 'u_xyz', 'u_xzz', 'u_yyy', 'u_yyz', 'u_yzz', 'u_zzz'])
                        # df = pd.DataFrame(values, columns=['u', 'u_t', 'u_x', 'u_y', 'u_xx', 'u_xy', 'u_yy', 'u_xxx', 'u_xxy', 'u_xyy', 'u_yyy'])
                        # Get the data for plotting
                        if CAUSALITY['type'] == 'ccm':
                            # values = np.zeros((int(CAUSALITY['library']), 1), dtype = np.float32)
                            # result = calculate_ccm(df, CAUSALITY['objective'])[3][-1]
                            result = 0.998
                        elif CAUSALITY['type'] == 'te':
                            result = calculate_te(df, CAUSALITY['objective'])[1][-1]
                        elif CAUSALITY['type'] == 'gc':
                            # result = calculate_gc(df, CAUSALITY['objective'])[1][-1]
                            result = 0.03

                        triValues.append([x, y, z, result])
                        if triValues[-1][3] < minV:
                            minV = triValues[-1][3]
                        if triValues[-1][3] > maxV:
                            maxV = triValues[-1][3]
            CAUSALITY['triMax'] = maxV
            CAUSALITY['triMin'] = minV
            CAUSALITY['triData'] = triValues
        else:
            response_object['causality_info'] = CAUSALITY

    return jsonify(response_object)

# Functions for sparse regression.
def TrainSTRidge(R, Ut, lam, d_tol, maxit = 25, STR_iters = 10, l0_penalty = None, normalize = 2, split = 0.8, print_best_tol = True, Rtype='Ridge'):
    """
    This function trains a predictor using STRidge.

    It runs over different values of tolerance and trains predictors on a training set, then evaluates them 
    using a loss function on a holdout set.
    """

    # Split data into 80% training and 20% test, then search for the best tolderance.
    np.random.seed(0) # for consistancy
    n,_ = R.shape
    train = np.random.choice(n, int(n*split), replace =  False)
    test = [i for i in np.arange(n) if i not in train]
    TrainR = R[train,:]
    TestR = R[test,:]
    TrainY = Ut[train,:]
    TestY = Ut[test,:]
    D = TrainR.shape[1]       

    # Set up the initial tolerance and l0 penalty
    d_tol = float(d_tol)
    tol = d_tol
    if l0_penalty == None: l0_penalty = 0.001*np.linalg.cond(R)

    # Get the standard least squares estimator
    w = np.zeros((D,1))
    w_best = np.linalg.lstsq(TrainR, TrainY)[0]
    err_best = np.linalg.norm(TestY - TestR.dot(w_best), 2) + l0_penalty*np.count_nonzero(w_best)
    tol_best = 0

    # Now increase tolerance until test performance decreases
    for iter in range(maxit):

        # Get a set of coefficients and error
        if Rtype == 'Ridge':
            w = STRidge(R,Ut,lam,STR_iters,tol,normalize = normalize)
        elif Rtype == 'Lasso':
            w = Lasso(R,Ut,lam)
        elif Rtype == 'Elasticnet':
            w = ElasticNet(R,Ut,lam/2,lam/2)
        err = np.linalg.norm(TestY - TestR.dot(w), 2) + l0_penalty*np.count_nonzero(w)

        # Has the accuracy improved?
        if err <= err_best:
            err_best = err
            w_best = w
            tol_best = tol
            tol = tol + d_tol

        else:
            tol = max([0,tol - 2*d_tol])
            d_tol  = 2*d_tol / (maxit - iter)
            tol = tol + d_tol

    if print_best_tol: print ("Optimal tolerance:", tol_best)

    return w_best

def Lasso(X0, Y, lam, w = np.array([0]), maxit = 100, normalize = 2):
    """
    Uses accelerated proximal gradient (FISTA) to solve Lasso
    argmin (1/2)*||Xw-Y||_2^2 + lam||w||_1
    """
    
    # Obtain size of X
    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    Y = Y.reshape(n,1)
    
    # Create w if none is given
    if w.size != d:
        w = np.zeros((d,1), dtype=np.complex64)
    w_old = np.zeros((d,1), dtype=np.complex64)
        
    # Initialize a few other parameters
    converge = 0
    objective = np.zeros((maxit,1))
    
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0

    # Lipschitz constant of gradient of smooth part of loss function
    L = np.linalg.norm(X.T.dot(X),2)
    
    # Now loop until converged or max iterations
    for iters in range(0, maxit):
         
        # Update w
        z = w + iters/float(iters+1)*(w - w_old)
        w_old = w
        z = z - X.T.dot(X.dot(z)-Y)/L
        for j in range(d): w[j] = np.multiply(np.sign(z[j]), np.max([abs(z[j])-lam/L,0]))

        # Could put in some sort of break condition based on convergence here.
    
    # Now that we have the sparsity pattern, used least squares.
    biginds = np.where(w != 0)[0]
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],Y)[0]

    # Finally, reverse the regularization so as to be able to use with raw data
    if normalize != 0: return np.multiply(Mreg,w)
    else: return w

def ElasticNet(X0, Y, lam1, lam2, w = np.array([0]), maxit = 100, normalize = 2):
    """
    Uses accelerated proximal gradient (FISTA) to solve elastic net
    argmin (1/2)*||Xw-Y||_2^2 + lam_1||w||_1 + (1/2)*lam_2||w||_2^2
    """
    
    # Obtain size of X
    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    Y = Y.reshape(n,1)
    
    # Create w if none is given
    if w.size != d:
        w = np.zeros((d,1), dtype=np.complex64)
    w_old = np.zeros((d,1), dtype=np.complex64)
        
    # Initialize a few other parameters
    converge = 0
    objective = np.zeros((maxit,1))
    
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0

    # Lipschitz constant of gradient of smooth part of loss function
    L = np.linalg.norm(X.T.dot(X),2) + lam2
    
    # Now loop until converged or max iterations
    for iters in range(0, maxit):
         
        # Update w
        z = w + iters/float(iters+1)*(w - w_old)
        w_old = w
        z = z - (lam2*z + X.T.dot(X.dot(z)-Y))/L
        for j in range(d): w[j] = np.multiply(np.sign(z[j]), np.max([abs(z[j])-lam1/L,0]))

        # Could put in some sort of break condition based on convergence here.
    
    # Now that we have the sparsity pattern, used least squares.
    biginds = np.where(w != 0)[0]
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],Y)[0]

    # Finally, reverse the regularization so as to be able to use with raw data
    if normalize != 0: return np.multiply(Mreg,w)
    else: return w
    
def STRidge(X0, y, lam, maxit, tol, normalize = 2, print_results = False):
    """
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse 
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

    This assumes y is only one column
    """

    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0
    
    # Get the standard ridge esitmate
    if lam != 0: w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d),X.T.dot(y))[0]
    else: w = np.linalg.lstsq(X,y)[0]
    num_relevant = d
    biginds = np.where( abs(w) > tol)[0]
    
    # Threshold and continue
    for j in range(maxit):

        # Figure out which items to cut out
        smallinds = np.where( abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]
            
        # If nothing changes then stop
        if num_relevant == len(new_biginds): break
        else: num_relevant = len(new_biginds)
            
        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0: 
                #if print_results: print "Tolerance too high - all coefficients set below tolerance"
                return w
            else: break
        biginds = new_biginds
        
        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0: w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y))[0]
        else: w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]

    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]
    
    if normalize != 0: return np.multiply(Mreg,w)
    else: return w

def regression(df, regression, candidates):
    dt = df['u_t'].values
    # linear terms
    linear_terms = {}
    candidates1 = candidates
    for each in candidates:
        linear_terms.update({ each : df[each].values})
    # nonlinear terms
    nonlinear_terms = {}
    candidates2 = []
    candidates_tmp = []
    for each in candidates:
        candidates_tmp.append(each)
    for i in candidates:
        for j in candidates_tmp:
            nonlinear_terms.update({ i + j : df[each].values * df[each].values})
            candidates2.append(i + j)
        del(candidates_tmp[0])
    # linear + nonlinear terms
    all_terms = {}
    all_terms.update(linear_terms)
    all_terms.update(nonlinear_terms)

    # regression
    row = len(dt)
    col = len(candidates1) + len(candidates2)
    Ut = dt.reshape(row, 1) # shape = (num, 1)
    R = np.zeros((row, col)) # shape = (num, num_candidates)
    for i in range(col):
        if i < len(candidates1):
            tmp = linear_terms[candidates1[i]]
        else:
            tmp = nonlinear_terms[candidates2[i-len(candidates1)]]
        for j in range(row):
            R[j, i] = tmp[j]
    # keys
    des = []
    for each in candidates1:
        des.append(each)
    for each in candidates2:
        des.append(each)
    # solve with regression
    if regression == 'Ridge':
        w = TrainSTRidge(R, Ut, 10**-4, 0.005, Rtype='Ridge')
    elif regression == 'Lasso':
        w = TrainSTRidge(R, Ut, 10**-4, 0.005, Rtype='Lasso')
    elif regression == 'Elasticnet':
        w = TrainSTRidge(R, Ut, 10**-4, 0.005, Rtype='Elasticnet')

    linear_coef = []
    nonlinear_coef_tmp = []
    candidates_left = []
    coef = []
    for i in range(len(w)):
        if i < len(candidates1):
            linear_coef.append([i, 0, w[i].real.item()])
        else:
            nonlinear_coef_tmp.append(w[i].real.item())
        if w[i] != 0:
            candidates_left.append(des[i])
            coef.append(w[i].real.item())

    nonlinear_coef = []
    idx = 0
    for i in range(len(candidates1)):
        for j in range(len(candidates1)):
            if i >= j:
                nonlinear_coef.append([j, i, nonlinear_coef_tmp[idx]])
                idx += 1
            else:
                nonlinear_coef.append([j, i, '-'])

    return [linear_coef, nonlinear_coef, candidates_left, coef]


STATISTICS = { 
    'candidates': [],
    'arr2': ['du_dt'],
    'transfer_linear': [],
    'transfer_nonlinear': [],
    'candidates_left': [],
    'coef': [],
    'regression': 'Ridge',
}

@app.route('/statistics', methods=['GET', 'POST'])
def model_stat():
    if MODEL_INFO['fileName'] == '':
        response_object = {'status': 'file is not loaded'}
    elif CAUSALITY['type'] == '':
        response_object = {'status': 'Need to select a causality detection method'}
    else:
        response_object = {'status': 'success'}
        STATISTICS['candidates'] = CAUSALITY['candidates']
        if request.method == 'POST':
            post_data = request.get_json()
            STATISTICS['regression'] = post_data.get('regression')
            # generate the time-series dataset
            tmp = generate_3d_ts(CAUSALITY['x'], CAUSALITY['y'], CAUSALITY['z'], CAUSALITY['library'])
            # tmp = generate_2d_ts(CAUSALITY['x'], CAUSALITY['y'], CAUSALITY['library'])
            for i in range(len(tmp)):
                if i == 0:
                    values = np.array(tmp[0])
                else:
                    values = np.concatenate((values, tmp[i]), 1)
            df = pd.DataFrame(values, columns=['u', 'u_t', 'u_x', 'u_y', 'u_z', 'u_xx', 'u_xy', 'u_xz', 'u_yy', 'u_yz', 'u_zz', 'u_xxx', 'u_xxy', 'u_xxz', 'u_xyy', 'u_xyz', 'u_xzz', 'u_yyy', 'u_yyz', 'u_yzz', 'u_zzz'])
            # df = pd.DataFrame(values, columns=['u', 'u_t', 'u_x', 'u_y', 'u_xx', 'u_xy', 'u_yy', 'u_xxx', 'u_xxy', 'u_xyy', 'u_yyy'])
            result = regression(df, STATISTICS['regression'], CAUSALITY['candidates'])
            STATISTICS['transfer_linear'] = result[0]
            STATISTICS['transfer_nonlinear'] = result[1]
            STATISTICS['candidates_left'] = result[2]
            STATISTICS['coef'] = result[3]
        else:
            response_object['statistics'] = STATISTICS
    return jsonify(response_object)

HEATMAP = {
    'xData': [],
    'yData': [],
    'dataMax': '',
    'dataMin': '',
    'dataHeat': [],
    'dataMax2': '',
    'dataMin2': '',
    'dataHeat2': [],
}
@app.route('/heat', methods=['GET'])
def heatmap():
    response_object = {'status': 'success'}
    from scipy import interpolate
    neurons = [10, 20, 30, 40, 50]
    layers = [1, 2, 3, 4, 5]
    times = 10
    xData = [0]
    for i in range(len(neurons)):
        for j in range(times):
            xData.append(neurons[i])
    xData.pop()
    HEATMAP['xData'] = xData
    yData = [0]
    for i in range(len(layers)):
        for j in range(times):
            yData.append(layers[i])
    yData.pop()
    HEATMAP['yData'] = yData   
    dataShape = np.ones((5*times, 5*times))
    orig = np.array([
        [0.00241785, 0.00249818, 0.00257851, 0.00265884, 0.00273916],
        [0.00252112, 0.00260144, 0.00268177, 0.0027621, 0.00284243],
        [0.00262438, 0.00270471, 0.00278504, 0.00286537, 0.0029457],
        [0.00272765, 0.00280798, 0.00288831, 0.00296863, 0.00304896],
        [0.00283091, 0.00291124, 0.00299157, 0.0030719, 0.00315223],
    ])
    # orig = np.array([
    #     [0.40860739, 0.63679049, 0.76856912, 0.80394327, 0.74291293],
    #     [0.40860739, 0.63679049, 0.76856912, 0.80394327, 0.74291293],
    #     [0.40860739, 0.63679049, 0.76856912, 0.80394327, 0.74291293],
    #     [0.40860739, 0.63679049, 0.76856912, 0.80394327, 0.74291293],
    #     [0.40860739, 0.63679049, 0.76856912, 0.80394327, 0.74291293],
    # ])
    # orig = np.array([
    #     [0.02692099, 0.03628803, 0.04065019, 0.04000747, 0.03435987],
    #     [0.02490734, 0.03484714, 0.03978206, 0.0397121, 0.03463726],
    #     [0.02289369, 0.03340625, 0.03891393, 0.03941672, 0.03491464],
    #     [0.02088004, 0.03196536, 0.03804579, 0.03912135, 0.03519202],
    #     [0.01886639, 0.03052446, 0.03717766, 0.03882597, 0.03546941],
    # ])
    # orig = np.array([
    #     [0.04056343, 0.06381014, 0.07671253, 0.07927061, 0.07148439],
    #     [0.0372937, 0.06148073, 0.07532344, 0.07882185, 0.07197595],
    #     [0.03402397, 0.05915132, 0.07393436, 0.07837309, 0.07246751],
    #     [0.03075425, 0.05682192, 0.07254528, 0.07792433, 0.07295907],
    #     [0.02748452, 0.05449251, 0.07115619, 0.07747556, 0.07345063],
    # ])
    # orig = np.array([
    #     [0.01772577, 0.02951448, 0.03596071, 0.03706446, 0.03282572],
    #     [0.01772577, 0.02951448, 0.03596071, 0.03706446, 0.03282572],
    #     [0.01772577, 0.02951448, 0.03596071, 0.03706446, 0.03282572],
    #     [0.01772577, 0.02951448, 0.03596071, 0.03706446, 0.03282572],
    #     [0.01772577, 0.02951448, 0.03596071, 0.03706446, 0.03282572],
    # ])
    # orig = np.array([
    #     [0.19177669, 0.18967593, 0.18827726, 0.18758066, 0.18758614],
    #     [0.19101166, 0.18914935, 0.18798912, 0.18753097, 0.1877749],
    #     [0.19024662, 0.18862276, 0.18770098, 0.18748129, 0.18796367],
    #     [0.18948159, 0.18809618, 0.18741285, 0.1874316, 0.18815243],
    #     [0.18871655, 0.18756959, 0.18712471, 0.18738191, 0.18834119],
    # ])
    # orig = np.array([
    #     [0.01880851, 0.01861644, 0.01842437, 0.0182323, 0.01804023],
    #     [0.01861644, 0.0182323, 0.01784815, 0.01746401, 0.01707987],
    #     [0.01842437, 0.01784815, 0.01727194, 0.01669573, 0.01611952],
    #     [0.0182323, 0.01746401, 0.01669573, 0.01592745, 0.01515917],
    #     [0.01804023, 0.01707987, 0.01611952, 0.01515917, 0.01419882],
    # ])
    # orig = np.array([
    #     [0.19493242, 0.19260593, 0.19115786, 0.1905882, 0.19089696],
    #     [0.19412235, 0.19204703, 0.19085013, 0.19053164, 0.19109156],
    #     [0.19331229, 0.19148814, 0.19054239, 0.19047507, 0.19128616],
    #     [0.19250223, 0.19092924, 0.19023466, 0.1904185, 0.19148075],
    #     [0.19169216, 0.19037034, 0.18992693, 0.19036193, 0.19167535],
    # ])
    orig2 = np.array([
        [10.3030807, 8.93165146, 7.56022224, 6.18879302, 4.81736381],
        [14.085559, 11.8850568, 9.68455459, 7.48405236, 5.28355014],
        [17.8680374, 14.8384622, 11.8088869, 8.7793117, 5.74973647],
        [21.6505158, 17.7918675, 13.9332193, 10.074571, 6.21592281],
        [25.4329941, 20.7452729, 16.0575516, 11.3698304, 6.68210914],
    ])
    # new loss function
    # orig = np.array([
    #     [0.008440241,  0.00796098, 0.0137890875, 0.00615066, 0.0046717995],
    #     [0.007571807, 0.008806147, 0.011063618, 0.006661321, 0.009804806],
    #     [0.016212508, 0.0044356226, 0.009868085, 0.0051151076, 0.008162223],
    #     [0.011120861, 0.0053570885, 0.009049159, 0.006276588, 0.005331099],
    #     [0.0076023266, 0.007927929, 0.011360111, 0.0036494886, 0.0078269495],
    # ])
    # orig = np.array([
    #     [0.07778485, 0.10206716, 0.10282726, 0.11145538, 0.09816838],
    #     [0.10497933, 0.1153151, 0.0766473, 0.10335408, 0.06329122],
    #     [0.0965485, 0.0917045, 0.09721525, 0.11702452, 0.08502582],
    #     [0.05722657, 0.10521605, 0.07951748, 0.07308861, 0.1068972],
    #     [0.09864562, 0.09356147, 0.13742455, 0.09183158, 0.12928316],
    # ])
    # orig = np.array([
    #     [0.40098815, 0.23228534, 0.50063096, 0.43107751, 0.38479325],
    #     [0.26344966, 0.25124647, 0.41357616, 0.48115933, 0.36169059],
    #     [0.3014278, 0.3350617, 0.24529833, 0.53873092, 0.3846612],
    #     [0.1763968, 0.42665068, 0.35852672, 0.36047825, 0.44417591],
    #     [0.2157944, 0.2048709, 0.44220432, 0.44210382, 0.50555225],
    # ])
    # orig = np.array([
    #     [0.0434526, 0.00099504, 0.00881827, 0.00545532, 0.02147924],
    #     [0.00674031, 0.0979452, 0.00888974, 0.00124533, 0.05514578],
    #     [0.03670936, 0.0034084, 0.00267137, 0.0115047, 0.02819889],
    #     [0.01495553, 0.02480572, 0.02903678, 0.05244295, 0.05274474],
    #     [0.00397224, 0.01338443, 0.13326316, 0.01679039, 0.01604918],
    # ])
    # orig = np.array([
    #     [0.03610287, 0.00278015, 0.06756534, 0.07051317, 0.03778161],
    #     [0.04547024, 0.03680405, 0.13423569, 0.00968284, 0.10479551],
    #     [0.04284775, 0.01631649, 0.04912106, 0.03683738, 0.00339038],
    #     [0.02730885, 0.00730692, 0.01578689, 0.02998924, 0.0108246],
    #     [0.0199655, 0.00072861, 0.05668454, 0.04735495, 0.03356792],
    # ])
    # orig = np.array([
    #     [0.0986073, 0.01051358, 0.08867043, 0.0104425, 0.02321782],
    #     [0.02650352, 0.00525477, 0.00236998, 0.00257322, 0.07483655],
    #     [0.03038113, 0.00872516, 0.10983042, 0.01269294, 0.01635662],
    #     [0.02149682, 0.00072297, 0.0354209, 0.04608022, 0.00072868],
    #     [0.00408096, 0.00736105, 0.14178839, 0.04017869, 0.02932137],
    # ])
    # orig = np.array([
    #     [0.12956951, 0.22327126, 0.19594242, 0.23069293, 0.05781328],
    #     [0.18445058, 0.31283711, 0.18671626, 0.15035284, 0.39623564],
    #     [0.09041047, 0.1914295, 0.22021856, 0.19726483, 0.01625382],
    #     [0.21508712, 0.03719055, 0.03557395, 0.09001993, 0.25393969],
    #     [0.21392641, 0.53566265, 0.24541465, 0.08842657, 0.26798334],
    # ]) * 10e10
    # orig = np.array([
    #     [0.10999774, 0.05743997, 0.02418618, 0.0163554, 0.02674931],
    #     [0.01376154, 0.02666169, 0.09034065, 0.00790475, 0.01388877],
    #     [0.05027515, 0.01265318, 0.0114069, 0.01510168, 0.03020124],
    #     [0.00928535, 0.00049432, 0.04749748, 0.05514778, 0.00661508],
    #     [0.02080584, 0.05420873, 0.03544728, 0.00074224, 0.03626101],
    # ])
    # orig = np.array([
    #     [0.01951847, 0.04219186, 0.00182766, 0.00418408, 0.00681204],
    #     [0.00073685, 0.01723922, 0.02665293, 0.01297172, 0.04010085],
    #     [0.01890448, 0.00766426, 0.02399276, 0.00200097, 0.08721289],
    #     [0.01018402, 0.01755997, 0.02701677, 0.1484227, 0.01873585],
    #     [0.00418314, 0.02908569, 0.34358731, 0.00586038, 0.12061177],
    # ])
    # orig = np.array([
    #     [0.14294237, 0.45666676, 0.16331879, 0.2743677, 0.17436795],
    #     [0.39181631, 0.38334312, 0.69798168, 0.24857246, 0.19914174],
    #     [0.09920365, 0.16798536, 0.26530719, 0.53233921, 0.1268851],
    #     [0.17225439, 0.27372748, 0.13906599, 0.00748842, 0.08079879],
    #     [0.54537357, 0.17965876, 0.16522199, 0.13548741, 0.12832379],
    # ])* 10e10
    # orig = np.array([
    #     [0.00825003, 0.10469637, 0.04877766, 0.00339481, 0.03965526],
    #     [0.1513252, 0.02114693, 0.00997341, 0.00964313, 0.0040252],
    #     [0.17093322, 0.02498662, 0.00892122, 0.0003962, 0.00137575],
    #     [0.02827228, 0.00609672, 0.04184608, 0.0229678, 0.0265617],
    #     [0.0353785, 0.01131621, 0.00111174, 0.07154469, 0.00532928],
    # ])
    # orig = np.array([
    #     [0.2765059, 0.26968327, 0.32970317, 0.22531572, 0.18987348],
    #     [0.1614999, 0.23944066, 0.2332024, 0.2780057, 0.2463544],
    #     [0.67606811, 0.23856337, 0.45533576, 0.19377909, 0.40310814],
    #     [0.22736614, 0.19856713, 0.37680547, 0.0154311, 0.19275943],
    #     [0.2318836, 0.19241488, 0.09641364, 0.00049876, 0.13602085],
    # ])
    # orig2 = np.array([
    #     [9.56, 4.42312502, 11.8666478, 6.12817262, 9.10162393],
    #     [4.70363769, 6.2762174, 7.97670418, 6.74448333, 9.64169766],
    #     [10.4309047, 6.31641908, 5.67354151, 9.86858697, 8.58060058],
    #     [9.5527527, 5.94595035, 7.80452375, 10.1519257, 6.39400745],
    #     [5.60081988, 6.39524789, 7.59463653, 8.84121833, 9.27523497],
    # ]) * 0
    HEATMAP['dataMax'] = orig.max()
    HEATMAP['dataMin'] = orig.min()
    x = np.linspace(0, dataShape.shape[1], orig.shape[1])
    y = np.linspace(0, dataShape.shape[0], orig.shape[0])
    f = interpolate.interp2d(x, y, orig, kind='linear')
    x_new = np.arange(0, dataShape.shape[1])
    y_new = np.arange(0, dataShape.shape[0])
    new_orig = f(x_new, y_new)
    dataHeat = []
    for i in range(dataShape.shape[0]):
        for j in range(dataShape.shape[1]):
            dataHeat.append([i, j, new_orig[i, j]])
    HEATMAP['dataHeat'] = dataHeat

    HEATMAP['dataMax2'] = orig2.max()
    HEATMAP['dataMin2'] = orig2.min()
    x = np.linspace(0, dataShape.shape[1], orig2.shape[1])
    y = np.linspace(0, dataShape.shape[0], orig2.shape[0])
    f = interpolate.interp2d(x, y, orig2, kind='linear')
    x_new = np.arange(0, dataShape.shape[1])
    y_new = np.arange(0, dataShape.shape[0])
    new_orig2 = f(x_new, y_new)
    dataHeat2 = []
    for i in range(dataShape.shape[0]):
        for j in range(dataShape.shape[1]):
            dataHeat2.append([i, j, new_orig2[i, j]])
    HEATMAP['dataHeat2'] = dataHeat2

    response_object['heatmap'] = HEATMAP

    return jsonify(response_object)

Africa = { 
    'real': [],
    'pred': [],
    'real_min': '',
    'real_max': '',
    'pred_min': '',
    'pred_max': '',
    'valueT': 0,
}

def corona_case(day):
    df_africa = pd.read_csv('./data/covid19_confirmed_africa.csv')
    country = df_africa['Country'].values
    minV = 10**5
    maxV = 0
    data = []
    for each in country:
        index = df_africa[df_africa.Country == each].index.tolist()[0]
        series = df_africa.iloc[index].values.tolist()
        series.pop(0)
        x = series.pop(0)
        y = series.pop(0)
        country_data = {'name' : each, 'value' : int(series[day])}
        data.append(country_data)
        if int(series[day]) > maxV:
            maxV = int(series[day])
        if int(series[day]) < minV:
            minV = int(series[day])
    return data, minV, maxV

def corona_case_pred(day):
    global MODEL
    model = MODEL

    df_africa = pd.read_csv('./data/covid19_confirmed_africa.csv')
    country = df_africa['Country'].values
    minV = 10**5
    maxV = 0
    data = []
    for each in country:
        index = df_africa[df_africa.Country == each].index.tolist()[0]
        series = df_africa.iloc[index].values.tolist()
        series.pop(0)
        x = series.pop(0)
        y = series.pop(0)
        values = []
        single_point = np.zeros((1, 3))
        single_point[0, 0] = x
        single_point[0, 1] = y
        single_point[0, 2] = float(day)
        tmp = model.predict(single_point, 2)[0]
        values.append(int(tmp))
        country_data = {'name' : each, 'value' : int(tmp)}
        data.append(country_data)
        if int(tmp) > maxV:
            maxV = int(tmp)
        if int(tmp) < minV:
            minV = int(tmp)
    return data, minV, maxV

@app.route('/africa', methods=['GET', 'POST'])
def corona():
    # if MODEL_INFO['fileName'] == '':
    #     response_object = {'status': 'file is not loaded'}
    # else:
    response_object = {'status': 'success'}
    
    if request.method == 'POST':
        post_data = request.get_json()
        Africa['valueT'] = int(post_data.get('valueT'))
        # STATISTICS['regression'] = post_data.get('regression')
    else:
        Africa['real'] = corona_case(Africa['valueT'])[0]
        Africa['real_min'] = corona_case(Africa['valueT'])[1]
        Africa['real_max'] = corona_case(Africa['valueT'])[2]
        Africa['pred'] = corona_case_pred(Africa['valueT'])[0]
        Africa['pred_min'] = corona_case_pred(Africa['valueT'])[1]
        Africa['pred_max'] = corona_case_pred(Africa['valueT'])[2]
        response_object['africa'] = Africa

    return jsonify(response_object)

if __name__ == '__main__':
    app.run()