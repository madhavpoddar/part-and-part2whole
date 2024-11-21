import pandas as pd
from bokeh.server.server import Server
# from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import column
from bokeh.palettes import Category10 as palette
from bokeh.io import show, export_png

from generate_vis import *

class p_and_p2w_widget():

    def __init__(self, df: pd.DataFrame):
        self.dict_p = {}
        self.df = df
        self.dict_p["arc_bar_chart"] = arc_bar_chart(df, "part", "whole", "id", "color")
        self.dict_p["scatterplot"] = scatterplot(df, "part", "whole", "id", "color")
        self.dict_p["progress_bar_charts"] = progress_bar_chart(df, "part", "whole", "id", "color")
        self.dict_p["bar_plus_angle_chart"] = bar_plus_angle_chart(df, "part", "whole", "id", "color")
        self.dict_p["T_bar_chart"] = T_bar_chart(df, "part", "whole", "id", "color")
        self.dict_p["glyph_pie_charts"] = glyph_pie_charts(df, "part", "whole", "id", "color")
        self.dict_p["glyph_ellipse_chart"] = glyph_ellipse_chart(df, "part", "whole", "id", "color")
        self.dict_p["glyph_bubble_gauge_charts"] = glyph_bubble_gauge_charts(df, "part", "whole", "id", "color")
        self.dict_p["glyph_keyhole_charts"] = glyph_keyhole_charts(df, "part", "whole", "id", "color")
        self.dict_p["circumference_point_aligned_keyhole_charts"] = circumference_point_aligned_keyhole_charts(df, "part", "whole", "id", "color")
        self.dict_p["overlapping_keyhole_charts"] = overlapping_keyhole_charts(df, "part", "whole", "id", "color")
        self.dict_p["bubbles_1d_chart"] = bubbles_1d_chart(df, "part", "whole", "id", "color")
        self.dict_p["circumference_point_aligned_bubble_gauge_charts"] = circumference_point_aligned_bubble_gauge_charts(df, "part", "whole", "id", "color")
        self.dict_p["glyph_inverted_T_bar_charts"] = glyph_inverted_T_bar_charts(df, "part", "whole", "id", "color")
        self.dict_p["pendulum_chart"] = pendulum_chart(df, "part", "whole", "id", "color")
        self.dict_p["overlapping_bubble_gauge_charts"] = overlapping_bubble_gauge_charts(df, "part", "whole", "id", "color")
        self.dict_p["radial_bar_chart_with_absolute_values_intercepts"] = radial_bar_chart_with_absolute_values_intercepts(df, "part", "whole", "id", "color")
        self.dict_p["spring_bar_chart"] = spring_bar_chart(df, "part", "whole", "id", "color")
        self.dict_p["overlapping_circular_area_chart"] = overlapping_circular_area_chart(df, "part", "whole", "id", "color")
        self.dict_p["glyph_wand_plus_charts"] = glyph_wand_plus_charts(df, "part", "whole", "id", "color")
        self.dict_p["overlapping_pie_chart"] = overlapping_pie_chart(df, "part", "whole", "id", "color")
        self.dict_p["radial_bar_chart"] = radial_bar_chart(df, "part", "whole", "id", "color")
        self.dict_p["two_level_treemap"] = two_level_treemap(df, "part", "whole", "id", "color")
        self.dict_p["two_level_icicle_plot"] = two_level_icicle_plot(df, "part", "whole", "id", "color")
        self.dict_p["two_level_parallel_set_plot"] = two_level_parallel_set_plot(df, "part", "whole", "id", "color")
        self.dict_p["marimekko_chart"] = marimekko_chart(df, "part", "whole", "id", "color")
        self.dict_p["inverted_triangle_chart"] = inverted_triangle_chart(df, "part", "whole", "id", "color")
        self.dict_p["inverted_triangle_chart_overlapping"] = inverted_triangle_chart_overlapping(df, "part", "whole", "id", "color")
        self.dict_p["glyph_donut_charts"] = glyph_donut_charts(df, "part", "whole", "id", "color")

    # def modify_doc(self, doc):
    
    def generate_layout(self):
        # Create the layout with the figures
        # layout = Tabs(tabs=[Panel(child=p, title=title) for (title, p) in self.dict_p.items()])
        layout = column([p for p in self.dict_p.values()])
        # for (title, p) in self.dict_p.items():
        #     export_png(obj=p,filename=title+".png")
        return layout

    def start_bokeh_server(self):
        #  Currently, not a server, but rather a standalone HTML
        show(self.generate_layout())
        
        # # Set up the Bokeh server application
        # server = Server({'/': self.modify_doc}, num_procs=1)
        # # Start the server
        # server.start()
        # # Open the application in a browser
        # server.show('/')
        # # Run the application
        # server.io_loop.start()



df = pd.DataFrame({"part": [77, 127, 42], "whole": [252, 195, 100]})

df["part-to-whole"]=df["part"]/df["whole"]
df["id"] = [str(i) for i in range(1, df.shape[0]+1)]
df["color"] = palette[len(df.index)]
print(df)

widget = p_and_p2w_widget(df)
widget.start_bokeh_server()