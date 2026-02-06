import numpy as np


def temp_based_split(X_ic_test, X_context_test, y_test, verbose=True):
    buckets = {
        '25': {'X_ic': [], 'X_ctx': [], 'y': []},
        '35': {'X_ic': [], 'X_ctx': [], 'y': []},
        '45': {'X_ic': [], 'X_ctx': [], 'y': []},
    }

    for i in range(X_context_test.shape[0]):
        if X_context_test[i, 1] > 0:  # temp_25 one-hot
            buckets['25']['X_ic'].append(X_ic_test[i])
            buckets['25']['X_ctx'].append(X_context_test[i])
            buckets['25']['y'].append(y_test[i])
        if X_context_test[i, 2] > 0:  # temp_35 one-hot
            buckets['35']['X_ic'].append(X_ic_test[i])
            buckets['35']['X_ctx'].append(X_context_test[i])
            buckets['35']['y'].append(y_test[i])
        if X_context_test[i, 3] > 0:  # temp_45 one-hot
            buckets['45']['X_ic'].append(X_ic_test[i])
            buckets['45']['X_ctx'].append(X_context_test[i])
            buckets['45']['y'].append(y_test[i])

    for t in ['25', '35', '45']:
        buckets[t]['X_ic'] = np.array(buckets[t]['X_ic'])
        buckets[t]['X_ctx'] = np.array(buckets[t]['X_ctx'])
        buckets[t]['y'] = np.array(buckets[t]['y'])

    if verbose:
        print('-----25 degree Celsius-----')
        print('X_ic shape:', buckets['25']['X_ic'].shape)
        print('X_context shape:', buckets['25']['X_ctx'].shape)
        print('y_test shape:', buckets['25']['y'].shape, '\n')

        print('-----35 degree Celsius-----')
        print('X_ic shape:', buckets['35']['X_ic'].shape)
        print('X_context shape:', buckets['35']['X_ctx'].shape)
        print('y_test shape:', buckets['35']['y'].shape, '\n')

        print('-----45 degree Celsius-----')
        print('X_ic shape:', buckets['45']['X_ic'].shape)
        print('X_context shape:', buckets['45']['X_ctx'].shape)
        print('y_test shape:', buckets['45']['y'].shape, '\n')

    return {
        '25': buckets['25'],
        '35': buckets['35'],
        '45': buckets['45'],
        'complete': {
            'X_ic': X_ic_test,
            'X_ctx': X_context_test,
            'y': y_test,
        },
    }
