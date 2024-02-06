import easygraph as eg

edge_list = [
    [[0,1,2],[3,4,5,6]],
    [[0,1,3],[4,5]],
    [[1,2,4],[1,2,5,6],[6,7,9]],
    [[1,3,5],[0,1]],
    [[1,2,8],[1,2,5,6,8],[6,7,8]],
    [[1,2],[1,4,7],[2,5,9],[1,3,5],[4,5]],
    [[1,2],[2,4,5]],
    [[1,4,7],[2,5,6,7,8,9],[3,5]],
    [[1,9],[3,4,7],[2,7,9],[1,2,5],[1,5]],
]
v_property_lst = ["Group1"]*2 + ["Group2"]*3 + ["Group3"]*3 + ["Group4"]*2
color_lst = ["#6e9ece"]*2 + ["#4e9595"]*3 + ["#e6928f"]*3 + ["#84574d"]*2
a = eg.Hypergraph(num_v=10)
a.add_hyperedges(edge_list[0],group_name="2022-01-01")
a.add_hyperedges(edge_list[1],group_name="2022-01-02")
a.add_hyperedges(edge_list[2],group_name="2022-01-03")
a.add_hyperedges(edge_list[3],group_name="2022-01-04")
a.add_hyperedges(edge_list[4],group_name="2022-01-05")
a.add_hyperedges(edge_list[5],group_name="2022-01-06")
a.add_hyperedges(edge_list[6],group_name="2022-01-07")
a.add_hyperedges(edge_list[7],group_name="2022-01-08")
a.add_hyperedges(edge_list[8],group_name="2022-01-09")

eg.draw_dynamic_hypergraph(a,
                           group_name_list=a.group_names,
                           save_path="dynamic_hypergraph.pdf",
                          v_line_width = 1,font_size=5,push_v_strength=50,
                        title_font_size=10,v_label=v_property_lst,
                           v_color=color_lst,e_color='red',e_line_width=1)



