# Helpers for adding convert in a graphical format
import ipywidgets as w
import scipp as sc
from scipp.plot import plot
import numpy as np
import IPython.display as disp


# Cheat dict to list allowed conversions. Should ideally get this from C++ code
allowed_conversions = {
    'tof': ('dspacing', 'wavelength', 'E'),
    'd-spacing': ('tof', ),
    'wavelength': ('Q', 'tof'),
    'E': ('tof', ),
    'Q': ('wavelength', ),
    '': tuple()
}

dimesion_to_unit = {
    'tof': sc.units.us,
    'd-spacing': sc.units.angstrom,
    'wavelength': sc.units.angstrom,
    'E': sc.units.meV,
    'Q': sc.units.one/sc.units.angstrom
}


class ConvertWidget(w.Box):
    def __init__(self, scope):
        super().__init__()
        self.scope = scope
        self.input = w.Text(placeholder='Input', value='', continuous_update=False)
        self.convert_from = w.Combobox(placeholder='from', disabled=False, continuous_update=False)
        self.convert_to = w.Combobox(placeholder='to', options=[], disabled=False, continuous_update=False)
        self.output = w.Text(placeholder='Output', value='', continuous_update=False)
        self.button = w.Button(description='convert')
        self.button.on_click(self._on_convert)
        self.input.observe(self._on_input_changed)
        self.convert_from.observe(self._on_from_changed)
        self.convert_to.observe(self._on_to_changed)
        self.children = [
            w.HBox([
                self.input, self.convert_from, self.convert_to, self.output, 
                self.button
            ])
        ]

        self.subscribers = []

    def subscribe(self, observer):
        self.subscribers.append(observer)

    def notify(self):
        for observer in self.subscribers:
            observer.update()

    def _on_convert(self, b):
        output_name = self.output.value
        input_name = self.input.value
        self.scope[output_name] = sc.neutron.convert(self.scope[input_name], self.convert_from.value, self.convert_to.value)
        self.notify()

    def _on_input_changed(self, change_dict):
        if change_dict['name'] == 'value':
            try:
                input = self.scope[change_dict['new']]
                self.convert_from.options = [key for key in allowed_conversions.keys() if key in input.coords]
            except KeyError:
                print(f'{change_dict["owner"].value} does not exist.')
            self.convert_from.disabled = False
            self.update_output()

    def _on_from_changed(self, change_dict):
        if change_dict['name'] == 'value':
            allowed_dimensions = change_dict['owner'].options if change_dict['owner'].options else allowed_conversions.keys()
            if change_dict['new'] not in allowed_dimensions:
                print(f"{change_dict['new']} not a recognised conversion dimension. Dimensions in data are {allowed_dimensions}")
                return
            self.convert_to.options = allowed_conversions[change_dict['new']]
            self.update_output()

    def _on_to_changed(self, change_dict):
        if change_dict['name'] == 'value':
            allowed_dimensions = change_dict['owner'].options if change_dict['owner'].options else allowed_conversions.keys()
            if change_dict['new'] not in allowed_dimensions:
                print(f"{change_dict['new']} not a recognised conversion dimension. Dimensions supported are {allowed_dimensions}")
                return
            self.update_output()

    def update_output(self, changes=None):
        if not self.output.value and self.convert_to.value and self.input.value:
            self.output.value = self.input.value + '_' + self.convert_to.value


class PlotWidget(w.Box):
    def __init__(self, scope):
        super().__init__()
        self.scope = scope
        options = [key for key, item in globals().items() if isinstance(item, (sc.DataArray, sc.Dataset))]
        self._data_selector = w.Combobox(placeholder='Data to plot', options=options)
        self._button = w.Button(description='Plot')
        self._button.on_click(self._on_button_clicked)
        self.plot_options = w.Output()#HTML()
        with self.plot_options:
            sc._repr_html_(self.scope)
        self.update_button = w.Button(description='Manual Update')
        self.update_button.on_click(self.update)
        self.output = w.Output(width='100%', height='100%')
        self.children = [w.VBox([w.HBox([self.plot_options, self.update_button]), w.HBox([self._data_selector, self._button]), self.output])]
        
    def _on_button_clicked(self, b):
        self.output.clear_output()
        with self.output:
            disp.display(plot(eval(self._data_selector.value, self.scope)))

    def update(self, b=None):
        self.plot_options.clear_output()
        with self.plot_options:
            sc._repr_html_(self.scope)


class DataCreationWidget(w.Box):
    def __init__(self, scope):
        super().__init__()
        self.name = w.Text(placeholder='name')
        self.dims = w.Combobox(placeholder='dims',
                               options=('tof',), ensure_option=False)
        self.num_spectra = w.Text(placeholder='num spectra', value='')
        self.button = w.Button(description='create')
        self.button.on_click(self._create_data)
        self.scope = scope
        self.children = [
            w.HBox([self.name, self.dims, self.num_spectra, self.button])
        ]
        self.subscribers = []

    def subscribe(self, observer):
        self.subscribers.append(observer)

    def notify(self):
        for observer in self.subscribers:
            observer.update()

    def _create_data(self, b):
        dim = self.dims.value
        if dim not in self.dims.options:
            print(f'Please enter a valid data dimension currently supported dimensions are {self.dims.options}')
            return
        num_spectra = int(self.num_spectra.value)
        output_name = self.name.value
        self.scope[output_name] = sc.Dataset(
            {
                'sample':
                sc.Variable(['spectrum', dim],
                            values=np.random.rand(num_spectra, 10),
                            variances=0.1 * np.random.rand(num_spectra, 10)),
                'background':
                sc.Variable(['spectrum', dim],
                            values=np.arange(0.0, num_spectra, 0.1).reshape(
                                num_spectra, 10),
                            variances=0.1 * np.random.rand(num_spectra, 10))
            },
            coords={
                dim:
                sc.Variable([dim], values=np.arange(11.0), unit=dimesion_to_unit[dim]),
                'spectrum':
                sc.Variable(['spectrum'],
                            values=np.arange(num_spectra),
                            unit=sc.units.one),
                'source-position': sc.Variable(value=np.array([1., 1., 10.]),
                           dtype=sc.dtype.vector_3_float64,
                           unit=sc.units.m),
                'sample-position': sc.Variable(value=np.array([1., 1., 60.]),
                           dtype=sc.dtype.vector_3_float64,
                           unit=sc.units.m),
                'position': sc.Variable(['spectrum'], values=np.arange(3*num_spectra).reshape(num_spectra, 3),
                            unit=sc.units.m, dtype=sc.dtype.vector_3_float64)
            })
        self.notify()
