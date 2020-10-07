import ipywidgets as w
import scipp as sc
from scipp.plot import plot
import numpy as np
import IPython.display as disp
import os

dimesion_to_unit = {
    'tof': sc.units.us,
    'd-spacing': sc.units.angstrom,
    'wavelength': sc.units.angstrom,
    'E': sc.units.meV,
    'Q': sc.units.one / sc.units.angstrom
}


def filepath_converter(filename):
    if filename.isdigit():
        # specified by run number
        filename = 'LARMOR' + filename

    # We will probably want to be a bit cleverer in how we hande file
    # finding and directory specifying. In particular browsing to files
    # or directories may be a requirment.
    directory = '/path/to/data/directory'
    filepath = os.path.join(directory, filename)
    # Commenting out as would always throw when loading fake files.
    # if not os.path.exists(filepath):
    #     raise ValueError(f'File {}')
    return filepath


class ProcessWidget(w.Box):
    def __init__(self, scope, callable, name, inputs, descriptions={}):
        super().__init__()
        self.scope = scope
        self.callable = callable

        self.input_widgets = []
        self.inputs = inputs
        self.setup_input_widgets(descriptions)

        self.output = w.Text(placeholder='Output',
                             value='',
                             continuous_update=False)

        self.button = w.Button(description=name)
        self.button.on_click(self._on_button_click)

        self.children = [
            w.HBox(self.input_widgets + [self.output, self.button])
        ]

        self.subscribers = []

    def setup_input_widgets(self, descriptions):
        for name in self.inputs.keys():
            placeholder = descriptions[name] if name in descriptions else name
            self.input_widgets.append(
                w.Text(placeholder=placeholder, continuous_update=False))

    def subscribe(self, observer):
        self.subscribers.append(observer)

    def notify(self):
        for observer in self.subscribers:
            observer.update()

    def _on_button_click(self, b):
        self.process()
        self.notify()

    def _retrive_kwargs(self):
        kwargs = {
            name: converter(item.value)
            for name, converter, item in zip(self.inputs.keys(
            ), self.inputs.values(), self.input_widgets)
        }
        return kwargs

    def process(self):
        try:
            kwargs = self._retrive_kwargs()
        except ValueError as e:
            print(f'Invalid inputs: {e}')
            return
        output_name = self.output.value
        self.scope[output_name] = self.callable(**kwargs)


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
        self.plot_options = w.Output()
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


def fake_load(filepath):
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


#Method to hide code blocks taken from
#https://stackoverflow.com/questions/27934885/how-to-hide-code-from-cells-in-ipython-notebook-visualized-with-nbviewer
javascript_functions = {False: "hide()", True: "show()"}
button_descriptions = {False: "Show code", True: "Hide code"}


def toggle_code(state):
    """
    Toggles the JavaScript show()/hide() function on the div.input element.
    """

    output_string = "<script>$(\"div.input\").{}</script>"
    output_args = (javascript_functions[state], )
    output = output_string.format(*output_args)

    disp.display(disp.HTML(output))


def button_action(value):
    """
    Calls the toggle_code function and updates the button description.
    """

    state = value.new

    toggle_code(state)

    value.owner.description = button_descriptions[state]


def setup_code_hiding():
    state = False
    toggle_code(state)

    button = w.ToggleButton(state, description=button_descriptions[state])
    button.observe(button_action, "value")
    return button
