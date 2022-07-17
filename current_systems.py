import numpy as np

east_halfdeg = [['FAC_in', [25, 22.5], 1e4],
                ['horizontal', [25, 22.5], [25, 23], 1e4],
                ['FAC_out', [25, 23], 1e4]]

east_1deg = [['FAC_in', [25, 22], 1e4],
             ['horizontal', [25, 22], [25, 23], 1e4],
             ['FAC_out', [25, 23], 1e4]]

east_3deg = [['FAC_in', [25, 21], 1e4],
             ['horizontal', [25, 21], [25, 24], 1e4],
             ['FAC_out', [25, 24], 1e4]]

east_6deg = [['FAC_in', [25, 19.5], 1e4],
             ['horizontal', [25, 19.5], [25, 25.5], 1e4],
             ['FAC_out', [25, 25.5], 1e4]]

east_9deg = [['FAC_in', [25, 18], 1e4],
             ['horizontal', [25, 18], [25, 27], 1e4],
             ['FAC_out', [25, 27], 1e4]]

east_12deg = [['FAC_in', [25, 16.5], 1e4],
              ['horizontal', [25, 16.5], [25, 28.5], 1e4],
              ['FAC_out', [25, 28.5], 1e4]]

east_15deg = [['FAC_in', [25, 15], 1e4],
              ['horizontal', [25, 15], [25, 30], 1e4],
              ['FAC_out', [25, 30], 1e4]]

south_halfdeg = [['FAC_in', [25, 22.5], 1e4],
                 ['horizontal', [25, 22.5], [25.5, 22.5], 1e4],
                 ['FAC_out', [25.5, 22.5], 1e4]]

south_1deg = [['FAC_in', [24.5, 22.5], 1e4],
              ['horizontal', [24.5, 22.5], [25.5, 22.5], 1e4],
              ['FAC_out', [25.5, 22.5], 1e4]]

south_2deg = [['FAC_in', [24, 22.5], 1e4],
              ['horizontal', [24, 22.5], [26, 22.5], 1e4],
              ['FAC_out', [26, 22.5], 1e4]]

south_4deg = [['FAC_in', [23, 22.5], 1e4],
              ['horizontal', [23, 22.5], [27, 22.5], 1e4],
              ['FAC_out', [27, 22.5], 1e4]]

south_6deg = [['FAC_in', [22, 22.5], 1e4],
              ['horizontal', [22, 22.5], [28, 22.5], 1e4],
              ['FAC_out', [28, 22.5], 1e4]]

south_8deg = [['FAC_in', [21, 22.5], 1e4],
              ['horizontal', [21, 22.5], [29, 22.5], 1e4],
              ['FAC_out', [29, 22.5], 1e4]]

south_10deg = [['FAC_in', [20, 22.5], 1e4],
               ['horizontal', [20, 22.5], [30, 22.5], 1e4],
               ['FAC_out', [30, 22.5], 1e4]]

eastwardstream = []
for colatitude in np.arange(24, 26.1, 0.5):
    eastwardstream.append(['FAC_in', [colatitude, 18], 1e4])
    eastwardstream.append(['horizontal', [colatitude, 18], [colatitude, 27], 1e4])
    eastwardstream.append(['FAC_out', [colatitude, 27], 1e4])

southwardstream = []
for longitude in np.arange(21.5, 23.6, 0.5):
    southwardstream.append(['FAC_in', [22, longitude], 1e4])
    southwardstream.append(['horizontal', [22, longitude], [28, longitude], 1e4])
    southwardstream.append(['FAC_out', [28, longitude], 1e4])

multidirection = [['FAC_in', [25, 29], 1e4],
                  ['horizontal', [25, 29], [21, 29], 5e3],
                  ['horizontal', [21, 29], [21, 20], 5e3],
                  ['horizontal', [21, 20], [25, 20], 5e3],
                  ['horizontal', [25, 29], [29, 29], 5e3],
                  ['horizontal', [29, 29], [29, 20], 5e3],
                  ['horizontal', [29, 20], [25, 20], 5e3],
                  ['FAC_in', [25, 16], 1e4],
                  ['horizontal', [25, 16], [25, 20], 1e4],
                  ['FAC_out', [25, 20], 2e4]]
