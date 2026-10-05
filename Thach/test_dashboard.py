import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Thach import dashboard as app
from Thach import dashboard_utils as du

def find_component(component, component_id):
    """Return the first Dash component with the requested id."""
    if getattr(component, "id", None) == component_id:
        return component
    children = getattr(component, "children", None)
    if children is None:
        return None
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        found = find_component(child, component_id)
        if found is not None:
            return found
    return None

def descendant_ids(component):
    ids = set()
    component_id = getattr(component, "id", None)
    if component_id:
        ids.add(component_id)
    children = getattr(component, "children", None)
    if children is None:
        return ids
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        ids.update(descendant_ids(child))
    return ids

class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fact, cls.bridges = app.FACT, app.BRIDGES
        cls.client = app.server.test_client()

    def test_coefficient_chart_uses_signed_vertical_bars(self):
        figure = app.coefficient_figure()
        traces = {trace.name: trace for trace in figure.data}
        self.assertEqual(set(traces), {"Hệ số dương (+)", "Hệ số âm (−)"})
        self.assertEqual(traces["Hệ số dương (+)"].marker.color, "#12a594")
        self.assertEqual(traces["Hệ số âm (−)"].marker.color, "#e5484d")
        negative = [value for value in traces["Hệ số âm (−)"].y if value is not None]
        positive = [value for value in traces["Hệ số dương (+)"].y if value is not None]
        self.assertEqual(len(negative), 2)
        self.assertTrue(np.all(np.asarray(negative) < 0))
        self.assertTrue(np.all(np.asarray(positive) >= 0))
        self.assertNotEqual(figure.layout.yaxis.zerolinewidth, 0)
        self.assertNotIn("?", figure.layout.title.text)
        prediction_panel = find_component(app.app.layout, "prediction-panel")
        self.assertNotIn("Cách đọc kết quả", str(prediction_panel))

    def callback(self, output_fragment, values, changed, state=None):
        key = next(key for key in app.app.callback_map if output_fragment in key)
        meta = app.app.callback_map[key]
        outputs = meta["output"]
        payload = [{"id": o.component_id, "property": o.component_property} for o in outputs] if isinstance(outputs, list) else {"id": outputs.component_id, "property": outputs.component_property}
        inputs = [{**item, "value": values.get(item["id"] + "." + item["property"])} for item in meta["inputs"]]
        states = [{**item, "value": (state or {}).get(item["id"] + "." + item["property"])} for item in meta["state"]]
        response = self.client.post("/_dash-update-component", json={"output": key, "outputs": payload, "inputs": inputs, "state": states, "changedPropIds": [changed]})
        self.assertEqual(response.status_code, 200, response.data[:1000])
        return response.get_json()["response"]

    def test_navigation_and_filters_are_in_expected_regions(self):
        sidebar = find_component(app.app.layout, "app-sidebar")
        main = find_component(app.app.layout, "dashboard-main")
        filters = find_component(app.app.layout, "filter-panel")
        self.assertIsNotNone(sidebar)
        self.assertIsNotNone(main)
        self.assertIsNotNone(filters)
        self.assertIsNone(find_component(app.app.layout, "page-intro"))
        self.assertIsNone(find_component(app.app.layout, "page-title"))
        self.assertIsNone(find_component(app.app.layout, "page-subtitle"))
        sidebar_ids = descendant_ids(sidebar)
        filter_ids = descendant_ids(filters)
        self.assertIn("hcmute-logo", sidebar_ids)
        self.assertIn("header-stats", sidebar_ids)
        self.assertIn("project-identity", sidebar_ids)
        self.assertNotIn("ĐỀ TÀI", str(sidebar))
        self.assertIn("DASHBOARD", str(sidebar))
        self.assertIn("COURSERA", str(sidebar))
        self.assertIn("page-tab", sidebar_ids)
        self.assertNotIn("filter-organizations", sidebar_ids)
        self.assertNotIn("breadcrumb", sidebar_ids)
        self.assertNotIn("clear-drill", sidebar_ids)
        self.assertNotIn("page-intro", descendant_ids(main))
        self.assertIn("filter-panel", descendant_ids(main))
        self.assertTrue({
            "filter-organizations", "filter-levels", "filter-subjects",
            "filter-skills", "filter-rating", "filter-hours",
            "filter-enrollment", "filter-options", "breadcrumb",
            "clear-drill", "reset-filters",
        }.issubset(filter_ids))

    def test_data_explorer_has_its_own_sidebar_tab(self):
        sidebar = find_component(app.app.layout, "app-sidebar")
        tabs = find_component(sidebar, "page-tab")
        data_panel = find_component(app.app.layout, "data-panel")
        self.assertEqual(
            [tab.value for tab in tabs.children],
            ["overview", "insight", "prediction", "data"],
        )
        self.assertEqual(
            [tab.label for tab in tabs.children],
            ["Tổng quan", "Phân tích", "Dự đoán người học", "Dữ liệu"],
        )
        self.assertIsNotNone(data_panel)
        self.assertTrue({
            "course-table", "course-detail",
        }.issubset(descendant_ids(data_panel)))
        self.assertNotIn("data-source-note", descendant_ids(data_panel))
        self.assertNotIn("coverage-note", descendant_ids(data_panel))
        self.assertIn("DASHBOARD", str(sidebar))
        self.assertIn("COURSERA", str(sidebar))
        self.assertNotIn("source-note", descendant_ids(sidebar))
        self.assertEqual(
            app.switch_tab("data"),
            (
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
                {"display": "block"},
                {"display": "block"},
            ),
        )
        self.assertEqual(app.switch_tab("prediction")[4], {"display": "none"})

    def test_default_preserves_all_courses_and_missing(self):
        filters = app.store_filters([], [], [], [], [0, 5], [0, app.HOURS_MAX], [0, app.ENROLL_MAX], ["missing"])
        filtered = du.filter_courses(self.fact, filters, self.bridges)
        self.assertEqual(len(filtered), len(self.fact))
        self.assertEqual(filtered.enrolled_num.isna().sum(), self.fact.enrolled_num.isna().sum())
        filters["include_missing"] = False
        filtered = du.filter_courses(self.fact, filters, self.bridges)
        self.assertFalse(filtered[["enrolled_num", "rating_num", "hours_to_complete"]].isna().any().any())

    def test_bridge_filters_do_not_multiply_courses(self):
        filters = {"subjects": ["business", "computer science"], "skills": ["data analysis"]}
        filtered = du.filter_courses(self.fact, filters, self.bridges)
        subject = self.bridges["subject"]
        skill = self.bridges["skill"]
        expected = set(subject.loc[subject.subject_normalized.isin(filters["subjects"]), "course_id"]) & set(skill.loc[skill.skill_normalized.isin(filters["skills"]), "course_id"])
        self.assertEqual(set(filtered.course_id), expected)
        self.assertEqual(len(filtered), len(expected))

    def test_reported_hours_and_strict_ranges(self):
        filtered = du.filter_courses(self.fact, {"reported_only": True, "hours": [5, 20], "include_missing": False}, self.bridges)
        self.assertGreater(len(filtered), 0)
        self.assertTrue(filtered.hours_to_complete.between(5, 20).all())
        self.assertTrue(filtered.hours_to_complete_is_estimated.eq(False).all())

    def test_pairwise_ranks_are_recomputed_after_missing(self):
        data = pd.DataFrame({"enrolled_num": [1, 2, 3, 4, 5], "num_reviews": [3, np.nan, 1, 2, np.nan], "rating_num": [4, 4, 4, 4, 4], "hours_to_complete": [1, 2, 3, 4, 5], "skill_count": [1, 2, 3, 4, 5]})
        rho, counts = du.pairwise_spearman(data)
        self.assertEqual(counts[0, 1], 3)
        self.assertAlmostEqual(rho[0, 1], -0.5)
        self.assertTrue(np.isnan(rho[0, 2]))

    def test_nine_chart_families_counts_and_ecdf(self):
        figures = du.build_figures(self.fact, self.bridges)
        expected = dict(level="bar", coverage="pie", histogram="histogram", rating="box", tree="treemap", map="choropleth", scatter="scattergl", heatmap="heatmap", ecdf="scatter")
        for key, kind in expected.items():
            self.assertEqual(figures[key].data[0].type, kind)
        self.assertEqual(sum(figures["level"].data[0].y), len(self.fact))
        self.assertEqual(sum(figures["coverage"].data[0].values), len(self.fact))
        self.assertEqual(sum(figures["map"].data[0].z), self.fact.organization_hq_country.isin(du.COUNTRY_ISO).sum())
        self.assertFalse(set(self.fact.organization_hq_country.dropna()) - set(du.COUNTRY_ISO))
        self.assertEqual(figures["ecdf"].data[0].y[-1], 1)
        self.assertTrue(np.all(np.diff(figures["ecdf"].data[0].y) >= 0))

    def test_every_chart_matches_its_source_rows(self):
        """Khóa số liệu hiển thị với fact/bridge để tránh biểu đồ đúng hình nhưng sai số."""
        figures = du.build_figures(self.fact, self.bridges)
        level_expected = self.fact.level_clean.fillna("Not specified").value_counts()
        level_trace = figures["level"].data[0]
        self.assertEqual(dict(zip(level_trace.x, level_trace.y)), level_expected.to_dict())
        status_order = ["reported_hours_and_weeks", "reported_hours_only", "estimated_from_months", "missing"]
        status_expected = self.fact.schedule_parse_status.fillna("missing").value_counts().reindex(status_order, fill_value=0)
        self.assertEqual(list(figures["coverage"].data[0].values), status_expected.tolist())
        enrolled = self.fact.loc[self.fact.enrolled_num.gt(0), "enrolled_num"]
        np.testing.assert_allclose(
            np.sort(figures["histogram"].data[0].x),
            np.sort(np.log10(enrolled)),
        )
        ecdf = figures["ecdf"].data[0]
        values, counts = np.unique(enrolled, return_counts=True)
        np.testing.assert_array_equal(ecdf.x, values)
        np.testing.assert_allclose(ecdf.y, np.cumsum(counts) / len(enrolled))
        rating_traces = figures["rating"].data
        self.assertEqual(sum(len(trace.y) for trace in rating_traces), self.fact.rating_num.count())
        for trace in rating_traces:
            level = trace.name.split(" (n=")[0]
            expected = self.fact.loc[
                self.fact.level_clean.eq(level) & self.fact.rating_num.notna(),
                "rating_num",
            ]
            np.testing.assert_array_equal(np.sort(trace.y), np.sort(expected))
        tree_trace = figures["tree"].data[0]
        tree_expected = (
            self.bridges["subject"]
            .groupby("subject_normalized").course_id.nunique()
            .sort_values(ascending=False)
            .head(20)
        )
        self.assertEqual(dict(zip(tree_trace.ids, tree_trace.values)), tree_expected.to_dict())
        mapped = self.fact[self.fact.organization_hq_country.isin(du.COUNTRY_ISO)]
        map_expected = mapped.groupby("organization_hq_country").course_id.nunique()
        map_trace = figures["map"].data[0]
        map_actual = dict(zip(map_trace.locations, map_trace.z))
        self.assertEqual(
            map_actual,
            {du.COUNTRY_ISO[country]: count for country, count in map_expected.items()},
        )
        scatter_ids = {
            course_id
            for trace in figures["scatter"].data
            for course_id in np.asarray(trace.customdata)[:, 0]
        }
        scatter_expected = set(self.fact.loc[
            self.fact.num_reviews.gt(0) & self.fact.enrolled_num.gt(0),
            "course_id",
        ])
        self.assertEqual(scatter_ids, scatter_expected)

        rho, counts = du.pairwise_spearman(self.fact)
        np.testing.assert_allclose(figures["heatmap"].data[0].z, rho, equal_nan=True)
        np.testing.assert_array_equal(figures["heatmap"].data[0].customdata, counts)
        organizations = app.du.insights.top_organizations(self.fact, 10)
        org_trace = figures["organizations"].data[0]
        self.assertEqual(list(org_trace.y), organizations.Organization.tolist())
        self.assertEqual(list(org_trace.x), organizations.n_courses.tolist())

    def test_chart_visual_contract_is_consistent(self):
        figures = du.build_figures(self.fact, self.bridges)
        self.assertEqual(app.GRAPH_CONFIG["topojsonURL"], "/assets/plotly-topojson/")
        for figure in figures.values():
            self.assertTrue(figure.layout.autosize)
            self.assertEqual(figure.layout.height, 400)
            self.assertTrue(figure.layout.title.text.startswith("<b>"))
        level = figures["level"].data[0]
        self.assertEqual(
            dict(zip(level.x, level.marker.color)),
            {name: du.LEVEL_COLORS[name] for name in level.x},
        )
        coverage = figures["coverage"].data[0]
        coverage_colors = dict(zip(coverage.labels, coverage.marker.colors))
        self.assertEqual(coverage_colors["Chưa có lịch học"], "#94a3b8")

        scatter_colors = {
            trace.name: trace.marker.color for trace in figures["scatter"].data
        }
        self.assertEqual(
            scatter_colors,
            {name: du.LEVEL_COLORS[name] for name in scatter_colors},
        )

        heatmap = figures["heatmap"].data[0]
        self.assertEqual((heatmap.zmin, heatmap.zmax), (-1, 1))
        colorbars = (
            figures["tree"].data[0].marker.colorbar,
            figures["map"].data[0].colorbar,
            heatmap.colorbar,
        )
        for colorbar in colorbars:
            self.assertLessEqual(colorbar.thickness, 12)

    def test_empty_and_single_row_figures(self):
        for frame in (self.fact.iloc[:0], self.fact.iloc[:1]):
            figures = du.build_figures(frame, self.bridges, "skill")
            self.assertEqual(len(figures), 10)
            for figure in figures.values():
                self.assertTrue(figure.to_json())

    def test_actual_callbacks_drill_filter_reset(self):
        values = {"chart-level.clickData": {"points": [{"x": "Beginner"}]}, "tree-kind.value": "subject"}
        response = self.callback("drill-state.data", values, "chart-level.clickData", {"drill-state.data": {}})
        drill = response["drill-state"]["data"]
        self.assertEqual(drill, {"level_clean": "Beginner"})
        for component, point, expected in (
            ("chart-map", {"location": "USA"}, {"organization_hq_country": "United States"}),
            ("chart-tree", {"id": "business"}, {"subject": "business"}),
            ("chart-organizations", {"y": "IBM"}, {"Organization": "IBM"}),
        ):
            response = self.callback("drill-state.data", {component + ".clickData": {"points": [point]}, "tree-kind.value": "subject"}, component + ".clickData", {"drill-state.data": {}})
            self.assertEqual(response["drill-state"]["data"], expected)
        response = self.callback("chart-level.figure", {"filter-state.data": {}, "drill-state.data": drill, "tree-kind.value": "subject"}, "drill-state.data")
        rows = response["course-table"]["data"]
        self.assertEqual(len(rows), self.fact.level_clean.eq("Beginner").sum())
        self.assertEqual({row["level_clean"] for row in rows}, {"Beginner"})
        response = self.callback("drill-state.data", {"reset-filters.n_clicks": 1}, "reset-filters.n_clicks", {"drill-state.data": drill})
        self.assertEqual(response["drill-state"]["data"], {})

    def test_detail_uses_stable_row_id_and_clears_on_filter(self):
        row = self.fact.iloc[100]
        values = {"course-table.active_cell": {"row": 0, "column": 0, "row_id": row.course_id}, "filter-state.data": {}, "drill-state.data": {}}
        response = self.callback("course-detail.children", values, "course-table.active_cell")
        self.assertIn(row.title, str(response))
        response = self.callback("course-detail.children", values, "filter-state.data")
        self.assertIsInstance(response["course-detail"]["children"], str)

    def test_empty_callback_prediction_and_routes(self):
        response = self.callback("chart-level.figure", {"filter-state.data": {"organizations": ["__empty__"]}, "drill-state.data": {}, "tree-kind.value": "subject"}, "filter-state.data")
        self.assertEqual(response["course-table"]["data"], [])
        result = app.update_model(1000, 4.6, 20, 6, "Beginner")
        self.assertIn("32,541", str(result))
        self.assertEqual(app.update_model(0, 4.6, 20, 6, "Beginner")[0], "Chưa thể ước lượng")
        for path in ("/", "/_dash-layout", "/_dash-dependencies", "/assets/dashboard.css",
                     "/assets/plotly-topojson/world_110m.json"):
            with self.client.get(path) as response:
                self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()
