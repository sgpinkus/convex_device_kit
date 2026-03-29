''' Replace make_home() to remove battery then reset agents field. '''
from device_kit import *
from device_kit.sample_scenarios.lcl.lcl_scenario import *


meta = {
  'title': 'The LCL reference scenario without batteries'
}


def make_home(type, id):
  return DeviceSet(id,
                   devices=[
      make_ac(type, id),
      make_phev(type, id),
      make_washer(type, id),
      make_lighting(type, id),
      make_entertainment(type, id),
    ],
    sbounds=(0, 100)
  )


# def matplot_network_writer_hook(event, plt, writer=None):
#   if event == 'after-update':
#     plt.title('')
#     plt.xlabel('Time (H)')
#     plt.ylabel('Power or Cost (kW or $)')
#   elif event == 'after-init':
#     writer.ymax = 15
#     writer.ymin = -15
