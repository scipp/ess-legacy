# Helpers for adding convert in a graphical format
import ipywidgets as w
import scipp as sc
from scipp.plot import plot
import numpy as np
import IPython.display as disp
from graphical_reduction import q1d, run_reduction, load_and_return

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
    'Q': sc.units.one / sc.units.angstrom
}


class ConvertWidget(w.Box):
    def __init__(self, scope):
        super().__init__()
        self.scope = scope
        self.input = w.Text(placeholder='Input',
                            value='',
                            continuous_update=False)
        self.convert_from = w.Combobox(placeholder='from',
                                       disabled=False,
                                       continuous_update=False)
        self.convert_to = w.Combobox(placeholder='to',
                                     options=[],
                                     disabled=False,
                                     continuous_update=False)
        self.output = w.Text(placeholder='Output',
                             value='',
                             continuous_update=False)
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
        self.scope[output_name] = sc.neutron.convert(self.scope[input_name],
                                                     self.convert_from.value,
                                                     self.convert_to.value)
        self.notify()

    def _on_input_changed(self, change_dict):
        if change_dict['name'] == 'value':
            try:
                input = self.scope[change_dict['new']]
                self.convert_from.options = [
                    key for key in allowed_conversions.keys()
                    if key in input.coords
                ]
            except KeyError:
                print(f'{change_dict["owner"].value} does not exist.')
            self.convert_from.disabled = False
            self.update_output()

    def _on_from_changed(self, change_dict):
        if change_dict['name'] == 'value':
            allowed_dimensions = change_dict['owner'].options if change_dict[
                'owner'].options else allowed_conversions.keys()
            if change_dict['new'] not in allowed_dimensions:
                print(
                    f"{change_dict['new']} not a recognised conversion dimension. Dimensions in data are {allowed_dimensions}"
                )
                return
            self.convert_to.options = allowed_conversions[change_dict['new']]
            self.update_output()

    def _on_to_changed(self, change_dict):
        if change_dict['name'] == 'value':
            allowed_dimensions = change_dict['owner'].options if change_dict[
                'owner'].options else allowed_conversions.keys()
            if change_dict['new'] not in allowed_dimensions:
                print(
                    f"{change_dict['new']} not a recognised conversion dimension. Dimensions supported are {allowed_dimensions}"
                )
                return
            self.update_output()

    def update_output(self, changes=None):
        if not self.output.value and self.convert_to.value and self.input.value:
            self.output.value = self.input.value + '_' + self.convert_to.value


class PlotWidget(w.Box):
    def __init__(self, scope):
        super().__init__()
        self.scope = scope
        options = [
            key for key, item in globals().items()
            if isinstance(item, (sc.DataArray, sc.Dataset))
        ]
        self._data_selector = w.Combobox(placeholder='Data to plot',
                                         options=options)
        self._button = w.Button(description='Plot')
        self._button.on_click(self._on_button_clicked)
        self.plot_options = w.Output()  #HTML()
        self.update_button = w.Button(description='Manual Update')
        self.update_button.on_click(self.update)
        self.output = w.Output(width='100%', height='100%')
        self.children = [
            w.VBox([
                w.HBox([self.plot_options, self.update_button]),
                w.HBox([self._data_selector, self._button]), self.output
            ])
        ]
        self.update()

    def _on_button_clicked(self, b):
        self.output.clear_output()
        with self.output:
            disp.display(plot(eval(self._data_selector.value, self.scope)))

    def update(self, b=None):
        options = [
            key for key, item in self.scope.items()
            if isinstance(item, (sc.DataArray, sc.Dataset, sc.Variable))
        ]
        self._data_selector.options = options
        self.plot_options.clear_output()
        with self.plot_options:
            sc._repr_html_(self.scope)


class DataCreationWidget(w.Box):
    def __init__(self, scope):
        super().__init__()
        self.name = w.Text(placeholder='name')
        self.dims = w.Combobox(placeholder='dims',
                               options=('tof', ),
                               ensure_option=False)
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
            print(
                f'Please enter a valid data dimension currently supported dimensions are {self.dims.options}'
            )
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
                sc.Variable([dim],
                            values=np.arange(11.0),
                            unit=dimesion_to_unit[dim]),
                'spectrum':
                sc.Variable(['spectrum'],
                            values=np.arange(num_spectra),
                            unit=sc.units.one),
                'source-position':
                sc.Variable(value=np.array([1., 1., 10.]),
                            dtype=sc.dtype.vector_3_float64,
                            unit=sc.units.m),
                'sample-position':
                sc.Variable(value=np.array([1., 1., 60.]),
                            dtype=sc.dtype.vector_3_float64,
                            unit=sc.units.m),
                'position':
                sc.Variable(['spectrum'],
                            values=np.arange(3 * num_spectra).reshape(
                                num_spectra, 3),
                            unit=sc.units.m,
                            dtype=sc.dtype.vector_3_float64)
            })
        self.notify()


class ReductionWidget(w.Box):
    def __init__(self, scope):
        super().__init__()
        self.sample = w.Text(description='Sample', value='49338')
        self.sample_trans = w.Text(description='Sample trans', value='49339')
        self.background = w.Text(description='Background', value='49334')
        self.background_trans = w.Text(description='Backg... trans',
                                       value='49335')
        self.load_button = w.Button(description='Load')
        self.load_button.on_click(self._on_load_button_clicked)
        self._button = w.Button(description='Process')
        self._button.on_click(self._on_process_button_clicked)
        self.output = w.Output(width='100%', height='100%')
        self.children = [
            w.VBox([
                w.HBox([self.sample, self.load_button]),
                w.HBox([self.sample_trans, self._button]), self.background,
                self.background_trans, self.output
            ])
        ]
        self.scope = scope
        self.subscribers = []

    def _on_process_button_clicked(self, b):
        self._on_load_button_clicked(None)
        sample = self.scope[self.data_name(self.sample.value)]
        sample_trans = self.scope[self.data_name(self.sample_trans.value)]
        background = self.scope[self.data_name(self.background.value)]
        background_trans = self.scope[self.data_name(self.background_trans.value)]
        moderator_file_path = f'{self.scope["path"]}/{self.scope["moderator_file"]}'
        direct_beam_file_path = f'{self.scope["path"]}/{self.scope["direct_beam_file"]}'
        l_collimation = self.scope['l_collimation']
        r1 = self.scope['r1']
        r2 = self.scope['r2']
        dr = self.scope['dr']
        wavelength_bins = self.scope['wavelength_bins']

        with self.output:
            reduced, sample_q1d, background_q1d = run_reduction(
                sample, sample_trans, background, background_trans,
                moderator_file_path, direct_beam_file_path, l_collimation, r1,
                r2, dr, wavelength_bins)
            
            #Need to think up a better naming scheme for reduced data
            reduced_name = f'reduced_sans{self.sample.value}'
            sample_name = f'sample_sans{self.sample.value}'
            background_name = f'background{self.background.value}'

            self.scope[reduced_name] = reduced
            self.scope[sample_name] = sample_q1d
            self.scope[background_name] = background_q1d
        self.notify()

    def _on_load_button_clicked(self, b):
        self.output.clear_output()
        run_list = [
            self.sample.value, self.sample_trans.value, self.background.value,
            self.background_trans.value
        ]
        run_list = [item for item in run_list if self.data_name(item) not in self.scope.keys()]
        with self.output:
            for run in run_list:
                data = load_and_return(run, self.scope['path'])
                name = self.data_name(run)
                self.scope[name] = data
        self.notify()

    def data_name(self, run):
        return self.scope['instrument'] + str(run)

    def subscribe(self, observer):
        self.subscribers.append(observer)

    def notify(self):
        for observer in self.subscribers:
            observer.update()
