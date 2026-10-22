# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

from unittest.mock import patch

import pytest

with patch("pydantic_ai.mcp.MCPServerSSE"):
    from agents.test_case_generation.main import (
        _chunk_by_sections,
        _classify_prd,
        _parse_sections_from_html,
        _parse_sections_from_text,
        _Section,
        _split_long_sections,
    )


# ---------------------------------------------------------------------------
# _classify_prd
# ---------------------------------------------------------------------------


class TestClassifyPrd:
    @pytest.mark.parametrize("heading", ["验收标准", "Acceptance Criteria"])
    def test_document_wording_does_not_select_the_strategy(self, heading):
        sections = "\n".join(f"5.{i} 节{i}" for i in range(1, 7))
        assert _classify_prd(f"{heading}\n{sections}") == "functional"
        assert _classify_prd(f"## {heading}\n1. 系统应...") == "narrative"

    def test_functional_numbered_sections(self):
        content = "\n".join(f"{i}.{j} 小节标题" for i in range(1, 4) for j in range(1, 4))
        assert _classify_prd(content) == "functional"

    def test_functional_threshold_exact(self):
        # Exactly at threshold (default 5)
        content = " ".join(f"{i}.{j}" for i in range(1, 3) for j in range(1, 4))  # 6 matches
        with patch("agents.test_case_generation.main.config") as mock_cfg:
            mock_cfg.PrdClassifierConfig.FUNCTIONAL_SECTION_THRESHOLD = 5
            assert _classify_prd(content) == "functional"

    def test_functional_below_threshold_is_narrative(self):
        content = "1.1 节一 1.2 节二"  # only 2 section numbers
        with patch("agents.test_case_generation.main.config") as mock_cfg:
            mock_cfg.PrdClassifierConfig.FUNCTIONAL_SECTION_THRESHOLD = 5
            assert _classify_prd(content) == "narrative"

    def test_narrative_plain_text(self):
        assert _classify_prd("这是一段没有结构的需求描述文字。") == "narrative"



# ---------------------------------------------------------------------------
# _parse_sections_from_html
# ---------------------------------------------------------------------------


class TestParseSectionsFromHtml:
    def test_minified_long_section_preserves_subsection_content(self):
        content = (
            '<h2>3.3 批量查询</h2><h4>3.3.1 请求参数</h4>'
            '<p>memberNo &amp; accountId</p><table><tr><td>参数约束</td></tr></table>'
            '<h4>3.3.2 返回结果</h4><p>告警标题与错误码</p>'
        )
        with patch("agents.test_case_generation.main.config") as cfg:
            cfg.PrdClassifierConfig.SECTION_MAX_CHARS = 20
            chunks = _chunk_by_sections(content)
        assert [item.id for item in chunks.items] == ["3.3.1", "3.3.2"]
        assert "memberNo & accountId" in chunks.items[0].text
        assert "参数约束" in chunks.items[0].text
        assert "告警标题与错误码" in chunks.items[1].text

    def test_basic_h2_parsing(self):
        html = '<h2 seq="5.1">5.1 背景</h2><p>背景内容</p><h2 seq="5.2">5.2 目标</h2><p>目标内容</p>'
        sections = _parse_sections_from_html(html)
        assert len(sections) == 2
        assert sections[0].section_id == "5.1"
        assert "背景内容" in sections[0].content

    def test_h3_parsing(self):
        html = "<h3>5.1.1 子节</h3><p>子节内容</p>"
        sections = _parse_sections_from_html(html)
        assert len(sections) == 1
        assert sections[0].section_id == "5.1.1"

    def test_no_headings_returns_empty(self):
        assert _parse_sections_from_html("<p>plain text</p>") == []

    def test_section_id_from_title_number(self):
        html = "<h2>3.2 用户故事</h2><p>内容</p>"
        sections = _parse_sections_from_html(html)
        assert sections[0].section_id == "3.2"

    def test_section_without_leading_number_gets_fallback_id(self):
        html = "<h2>简介</h2><p>内容</p>"
        sections = _parse_sections_from_html(html)
        assert sections[0].section_id == "sec-1"


# ---------------------------------------------------------------------------
# _parse_sections_from_text
# ---------------------------------------------------------------------------


class TestParseSectionsFromText:
    def test_numbered_sections(self):
        text = "5.1 节一\n内容一\n5.2 节二\n内容二"
        sections = _parse_sections_from_text(text)
        assert len(sections) == 2
        assert sections[0].section_id == "5.1"
        assert "内容一" in sections[0].content

    def test_no_numbered_sections(self):
        assert _parse_sections_from_text("plain content") == []


# ---------------------------------------------------------------------------
# _split_long_sections
# ---------------------------------------------------------------------------


class TestSplitLongSections:
    def test_numbered_requirements_without_following_body_are_retained(self):
        body = "3.3.1 请求必须包含 memberNo\n3.3.2 错误必须返回告警标题"
        sections = _split_long_sections([_Section("3.3", "批量查询", body)], max_chars=20)
        assert all(section.content for section in sections)
        assert "memberNo" in sections[0].content
        assert "告警标题" in sections[1].content

    def test_long_section_split_on_sub_headings(self):
        body = "前言内容\n5.1.1 子节一\n" + "x" * 500 + "\n5.1.2 子节二\n" + "y" * 500
        sections = [_Section("5.1", "节一", body)]
        result = _split_long_sections(sections, max_chars=200)
        ids = [s.section_id for s in result]
        assert "5.1.1" in ids
        assert "5.1.2" in ids

    def test_short_section_not_split(self):
        sections = [_Section("5.1", "节一", "a" * 100)]
        result = _split_long_sections(sections, max_chars=200)
        assert len(result) == 1

    def test_long_section_without_sub_headings_kept_as_is(self):
        sections = [_Section("5.1", "节一", "x" * 5000)]
        result = _split_long_sections(sections, max_chars=200)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _chunk_by_sections (integration)
# ---------------------------------------------------------------------------


class TestChunkBySections:
    def test_produces_non_empty_chunks_from_html(self):
        html = (
            "<h2>5.1 功能概述</h2><p>" + "内容 " * 60 + "</p>"
            "<h2>5.2 详细设计</h2><p>" + "详情 " * 60 + "</p>"
        )
        with patch("agents.test_case_generation.main.config") as mock_cfg:
            mock_cfg.PrdClassifierConfig.SECTION_MAX_CHARS = 5000
            result = _chunk_by_sections(html)
        assert len(result.items) >= 1
        assert all(item.source_type == "section" for item in result.items)

    def test_chunk_ids_match_section_numbers(self):
        html = "<h2>3.1 背景</h2><p>" + "x" * 100 + "</p>"
        with patch("agents.test_case_generation.main.config") as mock_cfg:
            mock_cfg.PrdClassifierConfig.SECTION_MAX_CHARS = 5000
            result = _chunk_by_sections(html)
        assert result.items[0].id == "3.1"

    def test_short_sections_are_separate_requirements(self):
        html = "<h2>5.1 退款</h2><p>全额退款撤销全部返现</p><h2>5.2 额度</h2><p>每月上限360</p>"
        result = _chunk_by_sections(html)
        assert [item.id for item in result.items] == ["5.1", "5.2"]

    def test_heading_only_parents_prefix_their_descendants(self):
        html = (
            "<h2>产品概述</h2><h3>1.1 活动目标</h3><p>目标</p>"
            "<h2>退款与撤销返现</h2><h3>5.1 冲正</h3><p>不返现</p><h3>5.2 退款</h3><p>撤销</p>"
        )
        result = _chunk_by_sections(html)
        assert [item.id for item in result.items] == ["1.1", "5.1", "5.2"]
        assert result.items[2].text.startswith("退款与撤销返现 / 5.2 退款")
        assert "撤销" in result.items[2].text

    def test_nested_heading_only_parents_stack_in_order(self):
        result = _chunk_by_sections("4.1 计算\n4.1.1 基数\n4.1.1.1 公式\n已结算金额减抵扣\n4.2 时间\nT+N")
        assert [item.id for item in result.items] == ["4.1.1.1", "4.2"]
        assert result.items[0].text.startswith("计算 / 基数 / 公式")
        assert result.items[1].text.startswith("时间")

    def test_body_written_inside_a_heading_is_kept(self):
        html = "<h3>6.2 已退回：原批次有效时恢复可用，已过期则立即失效</h3><h3>6.3 用户可见记录</h3><p>五类</p>"
        result = _chunk_by_sections(html)
        assert [item.id for item in result.items] == ["6.2", "6.3"]
        assert "已过期则立即失效" in result.items[0].text

    def test_duplicate_section_ids_are_made_unique(self):
        html = "<h2>简介</h2><p>a</p><h2>2.1 规则</h2><p>b</p><h2>2.1 规则</h2><p>c</p>"
        assert [item.id for item in _chunk_by_sections(html).items] == ["sec-1", "2.1", "2.1-2"]

    def test_scope_statements_in_the_document_do_not_change_requirements(self):
        html = "<h2>1 退款</h2><p>撤销返现</p><h2>2 额度</h2><p>每月上限</p><h2>验收标准</h2><p>本期只需验证 1</p>"
        without = _chunk_by_sections(html.replace("本期只需验证 1", "全部验证"))
        assert [i.id for i in _chunk_by_sections(html).items] == [i.id for i in without.items] == ["1", "2", "sec-3"]
        assert all(item.source_type == "section" for item in without.items)

    def test_fallback_to_text_parsing_when_no_html_headings(self):
        text = "5.1 节一\n" + "内容 " * 60 + "\n5.2 节二\n" + "内容 " * 60
        with patch("agents.test_case_generation.main.config") as mock_cfg:
            mock_cfg.PrdClassifierConfig.SECTION_MAX_CHARS = 5000
            result = _chunk_by_sections(text)
        assert len(result.items) >= 1
