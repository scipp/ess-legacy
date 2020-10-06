# Helpers for adding convert in a graphical format
import ipywidgets as w
import scipp as sc
from scipp.plot import plot
import numpy as np
import IPython.display as disp
from graphical_reduction import q1d, run_reduction, load_and_return
import os

# Cheat dict to list allowed conversions. Should ideally get this from C++ code
dimesion_to_unit = {
    'tof': sc.units.us,
    'd-spacing': sc.units.angstrom,
    'wavelength': sc.units.angstrom,
    'E': sc.units.meV,
    'Q': sc.units.one / sc.units.angstrom
}

class TransformWidget(w.Box):
    def __init__(self, scope, callable, name, inputs):
        super().__init__()
        self.scope = scope
        self.input_widgets = []
        self.input_converters = []
        self.callback = callable

        self.setup_inputs(inputs)
        self.output = w.Text(placeholder='Output',
                             value='',
                             continuous_update=False)

        self.output = w.Text(placeholder='Output',
                             value='',
                             continuous_update=False)
        self.button = w.Button(description=name)
        self.button.on_click(self._on_button_click)
        self.children = [
            w.HBox(self.input_widgets + [self.output, self.button])
        ]

        self.subscribers = []

    def setup_inputs(self, inputs):
        for name, converter in inputs.items():   
            self.input_widgets.append(
                w.Text(placeholder=name, continuous_update=False))
            self.input_converters.append(converter)

    def subscribe(self, observer):
        self.subscribers.append(observer)

    def notify(self):
        for observer in self.subscribers:
            observer.update()

    def _on_button_click(self, b):
        kwargs = {
            item.placeholder: converter(item.value)
            for item, converter in zip(self.input_widgets,
                                       self.input_converters)
        }
        output_name = self.output.value
        self.scope[output_name] = self.callback(**kwargs)
        self.notify()


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

    def _repr_html_(self, input_scope=None):
        import inspect
        # Is there a better way to get the scope? The `7` is hard-coded for the
        # current IPython stack when calling _repr_html_ so this is bound to break.
        scope = input_scope if input_scope else inspect.stack()[7][0].f_globals
        from IPython import get_ipython
        ipython = get_ipython()
        out = ''
        for category in ['Variable', 'DataArray', 'Dataset']:
            names = ipython.magic(f"who_ls {category}")
            out += f"<details open=\"open\"><summary>{category}s:"\
                   f"({len(names)})</summary>"
            for name in names:
                html = sc.table_html.make_html(eval(name, scope))
                out += f"<details style=\"padding-left:2em\"><summary>"\
                       f"{name}</summary>{html}</details>"
            out += "</details>"
        from IPython.core.display import display, HTML
        display(HTML(out))

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
            self._repr_html_(self.scope)


def fake_load(filename):
    dim = 'tof'
    num_spectra = 10
    return sc.Dataset(
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


class LoadWidget(w.Box):
    def __init__(
        self,
        scope,
        load_callable,
        directory,
        filename_descriptor,
        filename_converter,
        inputs = {}
    ):
        super().__init__()
        self.directory = directory
        self.filename = w.Text(placeholder=filename_descriptor)
        self.filename_converter = filename_converter

        self.input_widgets = []
        self.input_converters = []
        self.callback = load_callable

        self.setup_inputs(inputs)

        self.button = w.Button(description='Load')
        self.button.on_click(self._on_button_clicked)
        self.scope = scope
        self.children = [
            w.HBox([self.filename] + self.input_widgets + [self.button])
        ]
        self.subscribers = []

    def setup_inputs(self, inputs):
        for name, converter in inputs.items():   
            self.input_widgets.append(
                w.Text(placeholder=name, continuous_update=False))
            self.input_converters.append(converter)

    def subscribe(self, observer):
        self.subscribers.append(observer)

    def notify(self):
        for observer in self.subscribers:
            observer.update()

    def _on_button_clicked(self, b):
        kwargs = {
            item.placeholder: converter(item.value)
            for item, converter in zip(self.input_widgets,
                                       self.input_converters)
        }
        filename = self.filename_converter(self.filename.value)
        filepath = os.path.join(self.directory, filename)
        self.scope[filename] = self.callback(filepath, **kwargs)
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
        background_trans = self.scope[self.data_name(
            self.background_trans.value)]
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
        run_list = [
            item for item in run_list
            if self.data_name(item) not in self.scope.keys()
        ]
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
