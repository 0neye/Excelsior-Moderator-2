"""Restricted commands available only to moderators."""

from typing import TYPE_CHECKING, Any

import asyncio
from datetime import datetime, timedelta, timezone
import discord
from discord.ext import commands
from sqlalchemy import func

from config import ADMIN_ROLE, MODERATOR_ROLES
from database import FlaggedMessage, FlaggedMessageRating, MessageFeatures, RatingCategory
from utils import is_tracked_channel

if TYPE_CHECKING:
    from bot import ModerationResult


class ModFlagRecordSelect(discord.ui.Select):
    """Dropdown used to select a flagged message for moderation investigation."""

    def __init__(self, parent_view: "ModFlagInvestigationView"):
        """
        Build select options from the current recent flagged records.

        Args:
            parent_view: The owning view with recent flagged record data
        """
        self.parent_view = parent_view
        options = self._build_options()
        super().__init__(
            placeholder="Select a flagged message to inspect",
            min_values=1,
            max_values=1,
            options=options,
            row=0,
        )

    def _build_options(self) -> list[discord.SelectOption]:
        """
        Build select options for recent flagged records.

        Returns:
            List of Discord select options limited by API constraints
        """
        options: list[discord.SelectOption] = []
        for record_index, record in enumerate(self.parent_view.recent_records, start=1):
            author_id = record["author_id"]
            channel_id = record["channel_id"]
            waiver_suffix = " | waiver" if record["waiver_filtered"] else ""
            label = f"{record_index}. msg {record['message_id']}"
            description = f"<@{author_id}> in <#{channel_id}>{waiver_suffix}"
            options.append(
                discord.SelectOption(
                    label=label[:100],
                    description=description[:100],
                    value=str(record["message_id"]),
                )
            )
        return options

    async def callback(self, interaction: discord.Interaction):
        """Handle selection changes and refresh the active embed."""
        selected_value = str(self.values[0]) if self.values else ""
        if not selected_value.isdigit():
            await interaction.response.send_message(
                "Unable to parse the selected message ID.",
                ephemeral=True,
            )
            return
        self.parent_view.selected_message_id = int(selected_value)
        # Reset history-specific state when switching the selected flagged message
        self.parent_view.history_page_index = 0
        self.parent_view.selected_context_index = None
        if self.parent_view.mode == "history_detail":
            self.parent_view.mode = "history"
        self.parent_view._sync_button_states()

        embed = await self.parent_view.build_current_embed()
        await interaction.response.edit_message(embed=embed, view=self.parent_view)


class ModFlagContextSelect(discord.ui.Select):
    """Dropdown used to inspect one context message from the active history page."""

    def __init__(self, parent_view: "ModFlagInvestigationView"):
        """
        Initialize the context-message selector in a disabled placeholder state.

        Args:
            parent_view: The owning investigation view
        """
        self.parent_view = parent_view
        super().__init__(
            placeholder="Open History to load context messages",
            min_values=1,
            max_values=1,
            options=[discord.SelectOption(label="No context messages available", value="none")],
            disabled=True,
            row=1,
        )

    def refresh_options(self, context_messages: list[dict[str, Any]]) -> None:
        """
        Refresh dropdown options to match the currently visible history page.

        Args:
            context_messages: Full stored context list for the selected flagged message
        """
        page_entries = self.parent_view._get_history_page_entries(context_messages)
        if not page_entries:
            self.options = [discord.SelectOption(label="No context messages available", value="none")]
            self.placeholder = "No context messages available"
            self.disabled = True
            return

        option_rows: list[discord.SelectOption] = []
        for absolute_index, context_row in page_entries:
            author_name = self.parent_view._get_context_author_display(context_row)
            message_content = self.parent_view._sanitize_context_text(context_row.get("content") or "")
            compact_content = message_content[:70]
            if len(message_content) > 70:
                compact_content += "..."
            option_rows.append(
                discord.SelectOption(
                    label=f"#{absolute_index + 1} {author_name}"[:100],
                    description=(compact_content or "[No text content]")[:100],
                    value=str(absolute_index),
                )
            )

        self.options = option_rows
        self.placeholder = "Select a context message for detail view"
        self.disabled = self.parent_view.mode not in {"history", "history_detail"}

    async def callback(self, interaction: discord.Interaction):
        """Open detail mode for the selected context message row."""
        selected_value = str(self.values[0]) if self.values else ""
        if not selected_value.isdigit():
            await interaction.response.send_message(
                "Unable to parse the selected context row.",
                ephemeral=True,
            )
            return

        self.parent_view.selected_context_index = int(selected_value)
        self.parent_view.mode = "history_detail"
        self.parent_view._sync_button_states()
        embed = await self.parent_view.build_current_embed()
        await interaction.response.edit_message(embed=embed, view=self.parent_view)


class ModFlagInvestigationView(discord.ui.View):
    """Interactive moderator view for drilling into flagged-message decision details."""

    EMBED_FIELD_VALUE_LIMIT = 1024
    CODE_BLOCK_WRAPPER_LENGTH = 6
    HISTORY_PAGE_SIZE = 8
    HISTORY_EMBED_DESCRIPTION_LIMIT = 4096

    def __init__(
        self,
        cog: "Restricted",
        *,
        requesting_user_id: int,
        guild_id: int,
        recent_records: list[dict[str, Any]],
        selected_message_id: int | None = None,
    ):
        """
        Initialize investigation view state and UI components.

        Args:
            cog: Restricted cog providing data-fetch helper methods
            requesting_user_id: User ID allowed to interact with this view
            guild_id: Guild scope used for validation and links
            recent_records: Recent flagged records shown in the list panel
            selected_message_id: Optional initial selected message ID
        """
        super().__init__(timeout=600)
        self.cog = cog
        self.requesting_user_id = requesting_user_id
        self.guild_id = guild_id
        self.recent_records = recent_records
        self.selected_message_id = selected_message_id
        self.mode = "list"
        self.history_page_index = 0
        self.selected_context_index: int | None = None

        # Add a dropdown so moderators can pick a flagged message from recent rows
        self.add_item(ModFlagRecordSelect(self))
        # Add a second dropdown for selecting one context message in History mode
        self.add_item(ModFlagContextSelect(self))
        self._sync_button_states()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """
        Ensure only the invoking moderator can interact with this view.

        Args:
            interaction: Incoming interaction from a component action

        Returns:
            True when interaction should proceed, False otherwise
        """
        interaction_user = interaction.user
        if interaction_user is None:
            await interaction.response.send_message(
                "Unable to validate interaction user for this panel.",
                ephemeral=True,
            )
            return False
        if interaction_user.id != self.requesting_user_id:
            await interaction.response.send_message(
                "This investigation panel belongs to another moderator.",
                ephemeral=True,
            )
            return False
        return True

    async def on_timeout(self) -> None:
        """Disable all controls when the view times out."""
        for child in self.children:
            if isinstance(child, (discord.ui.Button, discord.ui.Select)):
                child.disabled = True

    def _sync_button_states(self) -> None:
        """Enable or disable controls based on selected record and active mode."""
        has_selection = self.selected_message_id is not None
        for child in self.children:
            if isinstance(child, ModFlagContextSelect):
                child.disabled = self.mode not in {"history", "history_detail"}
                continue
            if not isinstance(child, discord.ui.Button):
                continue
            if child.custom_id == "mod_flag_open_selected":
                child.disabled = not has_selection
            elif child.custom_id == "mod_flag_features":
                child.disabled = not has_selection
            elif child.custom_id == "mod_flag_history":
                child.disabled = not has_selection
            elif child.custom_id == "mod_flag_history_newer":
                child.disabled = not has_selection or self.mode not in {"history", "history_detail"}
            elif child.custom_id == "mod_flag_history_older":
                child.disabled = not has_selection or self.mode not in {"history", "history_detail"}
            elif child.custom_id == "mod_flag_back_to_list":
                child.disabled = self.mode == "list"

    def _get_context_select(self) -> ModFlagContextSelect | None:
        """
        Find the context selector component attached to this view.

        Returns:
            Context-select component when present, otherwise None
        """
        for child in self.children:
            if isinstance(child, ModFlagContextSelect):
                return child
        return None

    def _sanitize_context_text(self, raw_text: str) -> str:
        """
        Normalize context text so one-line previews stay readable.

        Args:
            raw_text: Source message content from stored context rows

        Returns:
            Single-line sanitized preview text
        """
        return raw_text.replace("`", "'").replace("\n", " ").strip()

    def _get_context_author_display(self, context_row: dict[str, Any]) -> str:
        """
        Build a display name for one context row author.

        Args:
            context_row: Serialized context row payload

        Returns:
            Human-friendly author display string
        """
        return context_row.get("author_name") or context_row.get("author_username") or "Unknown"

    def _get_history_total_pages(self, context_count: int) -> int:
        """
        Compute the page count for history browsing.

        Args:
            context_count: Total number of stored context messages

        Returns:
            Total history pages based on configured page size
        """
        if context_count <= 0:
            return 1
        return (context_count + self.HISTORY_PAGE_SIZE - 1) // self.HISTORY_PAGE_SIZE

    def _get_history_page_entries(
        self,
        context_messages: list[dict[str, Any]],
    ) -> list[tuple[int, dict[str, Any]]]:
        """
        Return context rows for the active page in chronological order.

        Args:
            context_messages: Full stored context list in chronological order

        Returns:
            List of (absolute_index, context_row) tuples for current page
        """
        total_count = len(context_messages)
        if total_count == 0:
            return []

        # Keep page 1 aligned with the oldest stored context rows for intuitive reading
        start_index = self.history_page_index * self.HISTORY_PAGE_SIZE
        end_index = min(start_index + self.HISTORY_PAGE_SIZE, total_count)
        return [
            (absolute_index, context_messages[absolute_index])
            for absolute_index in range(start_index, end_index)
        ]

    def _split_for_detail_fields(self, raw_text: str) -> tuple[list[str], bool]:
        """
        Split long context content into embed-safe detail chunks.

        Args:
            raw_text: Full context message content

        Returns:
            Tuple of (chunks, was_truncated)
        """
        max_inner_length = self.EMBED_FIELD_VALUE_LIMIT - self.CODE_BLOCK_WRAPPER_LENGTH
        chunk_size = min(900, max_inner_length)
        max_chunks = 4
        chunks = [
            raw_text[start_index:start_index + chunk_size]
            for start_index in range(0, len(raw_text), chunk_size)
        ]
        if not chunks:
            return ["[No text content]"], False

        was_truncated = len(chunks) > max_chunks
        visible_chunks = chunks[:max_chunks]
        if was_truncated and visible_chunks:
            # Reserve three characters for an ellipsis marker on the last visible chunk
            truncated_chunk = visible_chunks[-1]
            visible_chunks[-1] = f"{truncated_chunk[:-3]}..."
        return visible_chunks, was_truncated

    def _format_code_block_field_value(
        self,
        raw_text: str,
        *,
        empty_placeholder: str,
    ) -> str:
        """
        Format a Discord embed field value as a safe code block.

        Args:
            raw_text: Source text that should be displayed inside the code block
            empty_placeholder: Fallback text when source content is empty

        Returns:
            Code-block-wrapped field value guaranteed to fit embed limits
        """
        # Remove literal triple-backtick sequences so code fences cannot be broken
        sanitized_text = raw_text.replace("```", "'''")
        if not sanitized_text:
            sanitized_text = empty_placeholder

        # Keep the wrapped field under Discord's 1024-character embed field limit
        max_inner_length = self.EMBED_FIELD_VALUE_LIMIT - self.CODE_BLOCK_WRAPPER_LENGTH
        if len(sanitized_text) > max_inner_length:
            truncated_length = max_inner_length - 3
            sanitized_text = sanitized_text[:truncated_length] + "..."

        return f"```{sanitized_text}```"

    def _format_recent_line(self, record_index: int, record: dict[str, Any]) -> str:
        """
        Format one line in the recent flagged list.

        Args:
            record_index: Human-friendly record index in the current list
            record: Serialized recent flagged record payload

        Returns:
            Formatted markdown line for embed description
        """
        flagged_at = record["flagged_at"]
        is_selected = (
            self.selected_message_id is not None
            and record["message_id"] == self.selected_message_id
        )
        selected_marker = " <- selected" if is_selected else ""
        waiver_suffix = " (waiver filtered)" if record["waiver_filtered"] else ""
        return (
            f"{record_index}. <@{record['author_id']}> in <#{record['channel_id']}> at "
            f"<t:{int(flagged_at.timestamp())}:R> - "
            f"[Jump](https://discord.com/channels/{self.guild_id}/{record['channel_id']}/{record['message_id']})"
            f"{waiver_suffix}{selected_marker}"
        )

    async def build_current_embed(self) -> discord.Embed:
        """
        Build the current embed based on active mode and selected message.

        Returns:
            Embed configured for list, overview, feature, or history mode
        """
        if self.mode == "list":
            return self._build_recent_list_embed()

        if self.selected_message_id is None:
            return self._build_recent_list_embed()

        payload = self.cog._get_flagged_message_debug_payload(
            guild_id=self.guild_id,
            message_id=self.selected_message_id,
        )
        if payload is None:
            embed = discord.Embed(
                title="Flag Investigation",
                description="That flagged message could not be found in this server.",
                color=discord.Color.red(),
            )
            return embed

        if self.mode == "features":
            return self._build_features_embed(payload)
        if self.mode == "history_detail":
            return self._build_history_detail_embed(payload)
        if self.mode == "history":
            return self._build_history_embed(payload)
        return self._build_overview_embed(payload)

    def _build_recent_list_embed(self) -> discord.Embed:
        """
        Build embed listing recent flagged messages in this guild.

        Returns:
            List-view embed for selecting a flagged record
        """
        recent_lines = [
            self._format_recent_line(record_index, record)
            for record_index, record in enumerate(self.recent_records, start=1)
        ]
        embed = discord.Embed(
            title=f"Flag Investigation Queue ({len(self.recent_records)})",
            description="\n".join(recent_lines),
            color=discord.Color.orange(),
        )
        embed.add_field(
            name="How to Use",
            value=(
                "Pick a message from the dropdown, then click **Open Selected**. "
                "Use **Features** and **History** to drill into the moderation decision."
            ),
            inline=False,
        )
        return embed

    def _build_overview_embed(self, payload: dict[str, Any]) -> discord.Embed:
        """
        Build high-level overview for one flagged message.

        Args:
            payload: Serialized debug payload for one flagged message

        Returns:
            Overview embed containing metadata and rating summary
        """
        jump_url = (
            f"https://discord.com/channels/{self.guild_id}/"
            f"{payload['channel_id']}/{payload['message_id']}"
        )
        content_preview = payload["content"]

        embed = discord.Embed(
            title=f"Flag Overview: {payload['message_id']}",
            description=(
                f"**Author:** <@{payload['author_id']}>\n"
                f"**Target:** {payload['target_display']}\n"
                f"**Flagged:** <t:{int(payload['flagged_at'].timestamp())}:F>\n"
                f"**Action Taken:** {'Yes' if payload['was_acted_upon'] else 'No'}\n"
                f"**Waiver Filtered:** {'Yes' if payload['waiver_filtered'] else 'No'}\n"
                f"**Jump:** [Open message]({jump_url})"
            ),
            color=discord.Color.blurple(),
        )
        embed.add_field(
            name="Flagged Message Content",
            value=self._format_code_block_field_value(
                content_preview,
                empty_placeholder="[No text content]",
            ),
            inline=False,
        )

        ratings_summary = payload["ratings_summary"]
        embed.add_field(
            name="Human Ratings",
            value=ratings_summary,
            inline=False,
        )
        embed.add_field(
            name="Stored Feature Rows",
            value=str(payload["feature_row_count"]),
            inline=True,
        )
        embed.add_field(
            name="Stored Context Messages",
            value=str(payload["context_count"]),
            inline=True,
        )
        return embed

    def _build_features_embed(self, payload: dict[str, Any]) -> discord.Embed:
        """
        Build feature-value embed for one flagged message.

        Args:
            payload: Serialized debug payload including selected feature row

        Returns:
            Feature drill-down embed
        """
        selected_feature_row = payload["selected_feature_row"]
        if selected_feature_row is None:
            feature_text = "No runtime or extraction feature row found for this message."
            row_meta = "Unavailable"
        else:
            features = selected_feature_row.get("features", {})
            sorted_feature_items = sorted(features.items(), key=lambda item: item[0])
            feature_lines = [
                f"{feature_name}: {float(feature_value):.4f}"
                for feature_name, feature_value in sorted_feature_items
            ]
            feature_text = "\n".join(feature_lines) if feature_lines else "No features found."
            row_meta = (
                f"run_index={selected_feature_row.get('run_index', 0)} | "
                f"extraction_run_id={selected_feature_row.get('extraction_run_id')}"
            )

        embed = discord.Embed(
            title=f"Flag Features: {payload['message_id']}",
            description=(
                "Feature values used for model inference (or latest available extraction row).\n"
                f"**Selected Row:** {row_meta}"
            ),
            color=discord.Color.teal(),
        )
        embed.add_field(
            name="Feature Values",
            value=self._format_code_block_field_value(
                feature_text,
                empty_placeholder="No features found.",
            ),
            inline=False,
        )
        return embed

    def _build_history_embed(self, payload: dict[str, Any]) -> discord.Embed:
        """
        Build context-history embed for one flagged message.

        Args:
            payload: Serialized debug payload with context message list

        Returns:
            Context history drill-down embed
        """
        context_messages = payload["context_messages"]
        total_count = len(context_messages)
        total_pages = self._get_history_total_pages(total_count)
        if self.history_page_index >= total_pages:
            self.history_page_index = max(total_pages - 1, 0)

        # Keep the context-row dropdown aligned to the currently active page
        context_select = self._get_context_select()
        if context_select is not None:
            context_select.refresh_options(context_messages)

        if not context_messages:
            history_lines = ["No context messages were stored for this flagged record."]
        else:
            page_entries = self._get_history_page_entries(context_messages)
            history_lines = []
            for absolute_index, context_row in page_entries:
                author_name = self._get_context_author_display(context_row)
                message_content = self._sanitize_context_text(context_row.get("content") or "")
                compact_content = message_content[:90]
                if len(message_content) > 90:
                    compact_content += "..."
                history_lines.append(
                    f"**#{absolute_index + 1}** - {author_name}: {compact_content or '[No text content]'}"
                )

        history_description = "\n".join(history_lines)
        if len(history_description) > self.HISTORY_EMBED_DESCRIPTION_LIMIT:
            history_description = history_description[:self.HISTORY_EMBED_DESCRIPTION_LIMIT - 3] + "..."

        embed = discord.Embed(
            title=f"Flag Context History: {payload['message_id']}",
            description=history_description,
            color=discord.Color.green(),
        )
        embed.add_field(
            name="How to Use",
            value=(
                "Use **History Newer** and **History Older** to switch pages, "
                "then choose a row in the context dropdown for full detail."
            ),
            inline=False,
        )
        embed.set_footer(
            text=(
                f"Page {self.history_page_index + 1}/{total_pages} | "
                f"{payload['context_count']} total context messages"
            )
        )
        return embed

    def _build_history_detail_embed(self, payload: dict[str, Any]) -> discord.Embed:
        """
        Build detail view for one selected context message row.

        Args:
            payload: Serialized debug payload with context message list

        Returns:
            Context detail embed for one selected row
        """
        context_messages = payload["context_messages"]
        context_select = self._get_context_select()
        if context_select is not None:
            context_select.refresh_options(context_messages)

        if not context_messages:
            self.mode = "history"
            return self._build_history_embed(payload)

        if self.selected_context_index is None or not (
            0 <= self.selected_context_index < len(context_messages)
        ):
            self.selected_context_index = len(context_messages) - 1

        selected_row = context_messages[self.selected_context_index]
        author_name = self._get_context_author_display(selected_row)
        raw_content = (selected_row.get("content") or "").replace("```", "'''")
        content_chunks, was_truncated = self._split_for_detail_fields(raw_content)

        embed = discord.Embed(
            title=f"Flag Context Detail: {payload['message_id']}",
            description=(
                f"**Context Row:** #{self.selected_context_index + 1}\n"
                f"**Author:** {author_name}\n"
                "Use **History** to return to the paged list."
            ),
            color=discord.Color.green(),
        )

        # Show chunked content so one long context message cannot break field limits
        for chunk_index, chunk_text in enumerate(content_chunks, start=1):
            embed.add_field(
                name=f"Content Part {chunk_index}",
                value=self._format_code_block_field_value(
                    chunk_text,
                    empty_placeholder="[No text content]",
                ),
                inline=False,
            )

        if was_truncated:
            embed.add_field(
                name="Content Truncated",
                value=(
                    "This context message exceeded the detail panel limit and was truncated. "
                    "If needed, inspect the source record directly in storage."
                ),
                inline=False,
            )
        return embed

    @discord.ui.button(
        label="Open Selected",
        style=discord.ButtonStyle.primary,
        custom_id="mod_flag_open_selected",
        row=2,
    )
    async def open_selected_button(
        self,
        _button: discord.ui.Button,
        interaction: discord.Interaction,
    ):
        """Open overview mode for the currently selected flagged message."""
        self.mode = "overview"
        self._sync_button_states()
        embed = await self.build_current_embed()
        await interaction.response.edit_message(embed=embed, view=self)

    @discord.ui.button(
        label="Features",
        style=discord.ButtonStyle.secondary,
        custom_id="mod_flag_features",
        row=2,
    )
    async def features_button(
        self,
        _button: discord.ui.Button,
        interaction: discord.Interaction,
    ):
        """Open the feature-value drill-down for the selected flagged message."""
        self.mode = "features"
        self._sync_button_states()
        embed = await self.build_current_embed()
        await interaction.response.edit_message(embed=embed, view=self)

    @discord.ui.button(
        label="History",
        style=discord.ButtonStyle.secondary,
        custom_id="mod_flag_history",
        row=2,
    )
    async def history_button(
        self,
        _button: discord.ui.Button,
        interaction: discord.Interaction,
    ):
        """Open the context-history drill-down for the selected flagged message."""
        self.mode = "history"
        self.history_page_index = 0
        self.selected_context_index = None
        self._sync_button_states()
        embed = await self.build_current_embed()
        await interaction.response.edit_message(embed=embed, view=self)

    @discord.ui.button(
        label="History Newer",
        style=discord.ButtonStyle.secondary,
        custom_id="mod_flag_history_newer",
        row=3,
    )
    async def history_newer_button(
        self,
        _button: discord.ui.Button,
        interaction: discord.Interaction,
    ):
        """Move to a newer history page for the selected flagged message."""
        if self.selected_message_id is None:
            await interaction.response.send_message(
                "Select a flagged message first.",
                ephemeral=True,
            )
            return

        payload = self.cog._get_flagged_message_debug_payload(
            guild_id=self.guild_id,
            message_id=self.selected_message_id,
        )
        if payload is None:
            await interaction.response.send_message(
                "That flagged message could not be found in this server.",
                ephemeral=True,
            )
            return

        total_pages = self._get_history_total_pages(len(payload["context_messages"]))
        if self.history_page_index >= total_pages - 1:
            await interaction.response.send_message(
                "You are already on the newest history page.",
                ephemeral=True,
            )
            return

        # Newer pages are higher index values when page 1 starts at oldest rows
        self.history_page_index += 1
        self.selected_context_index = None
        self.mode = "history"
        self._sync_button_states()
        embed = await self.build_current_embed()
        await interaction.response.edit_message(embed=embed, view=self)

    @discord.ui.button(
        label="History Older",
        style=discord.ButtonStyle.secondary,
        custom_id="mod_flag_history_older",
        row=3,
    )
    async def history_older_button(
        self,
        _button: discord.ui.Button,
        interaction: discord.Interaction,
    ):
        """Move to an older history page for the selected flagged message."""
        if self.selected_message_id is None:
            await interaction.response.send_message(
                "Select a flagged message first.",
                ephemeral=True,
            )
            return

        payload = self.cog._get_flagged_message_debug_payload(
            guild_id=self.guild_id,
            message_id=self.selected_message_id,
        )
        if payload is None:
            await interaction.response.send_message(
                "That flagged message could not be found in this server.",
                ephemeral=True,
            )
            return

        if self.history_page_index <= 0:
            await interaction.response.send_message(
                "You are already on the oldest history page.",
                ephemeral=True,
            )
            return

        # Older pages are lower index values when page 1 starts at oldest rows
        self.history_page_index -= 1
        self.selected_context_index = None
        self.mode = "history"
        self._sync_button_states()
        embed = await self.build_current_embed()
        await interaction.response.edit_message(embed=embed, view=self)

    @discord.ui.button(
        label="Back to List",
        style=discord.ButtonStyle.gray,
        custom_id="mod_flag_back_to_list",
        row=3,
    )
    async def back_to_list_button(
        self,
        _button: discord.ui.Button,
        interaction: discord.Interaction,
    ):
        """Return to the recent flagged list panel."""
        self.mode = "list"
        self._sync_button_states()
        embed = await self.build_current_embed()
        await interaction.response.edit_message(embed=embed, view=self)


def is_moderator():
    """Check if the user has one of the moderator roles."""
    async def predicate(ctx: discord.ApplicationContext) -> bool:
        if not ctx.guild or not isinstance(ctx.author, discord.Member):
            return False
        
        # Get the member's role names
        member_role_names = [role.name for role in ctx.author.roles]
        
        # Check if any moderator role is in the member's roles
        return any(role in member_role_names for role in MODERATOR_ROLES)
    
    return commands.check(predicate)  # type: ignore[arg-type]


def is_admin():
    """Check if the user has the admin role."""
    async def predicate(ctx: discord.ApplicationContext) -> bool:
        if not ctx.guild or not isinstance(ctx.author, discord.Member):
            return False
        
        member_role_names = [role.name for role in ctx.author.roles]
        return ADMIN_ROLE in member_role_names
    
    return commands.check(predicate)  # type: ignore[arg-type]


class Restricted(commands.Cog):
    """Cog containing commands restricted to moderators."""
    
    def __init__(self, bot):
        self.bot = bot
        # Access to global message store and DB connection
        self.message_store = bot.message_store
        self.get_db_session = bot.get_db_session

    def _serialize_recent_flagged_records(
        self,
        guild_id: int,
        limit: int,
        include_waiver_filtered: bool,
    ) -> list[dict[str, Any]]:
        """
        Fetch recent flagged rows with fields needed for investigation UI.

        Args:
            guild_id: Guild ID used to scope rows to the active server
            limit: Maximum number of recent rows to fetch
            include_waiver_filtered: Whether waiver-filtered rows should be included

        Returns:
            List of serialized flagged-record dictionaries, newest first
        """
        session = self.get_db_session()
        try:
            query = session.query(FlaggedMessage).filter(FlaggedMessage.guild_id == guild_id)
            if not include_waiver_filtered:
                query = query.filter(FlaggedMessage.waiver_filtered.is_(False))
            rows = query.order_by(FlaggedMessage.flagged_at.desc()).limit(limit).all()
        finally:
            session.close()

        return [
            {
                "message_id": row.message_id,
                "channel_id": row.channel_id,
                "author_id": row.author_id,
                "flagged_at": row.flagged_at,
                "waiver_filtered": bool(row.waiver_filtered),
            }
            for row in rows
        ]

    def _summarize_ratings_for_flagged_message(self, ratings: list[FlaggedMessageRating]) -> str:
        """
        Build a compact ratings summary string grouped by category.

        Args:
            ratings: Rating rows associated with one flagged message

        Returns:
            Human-readable summary string suitable for an embed field
        """
        if not ratings:
            return "No moderator ratings recorded yet."

        category_counts: dict[str, int] = {}
        for rating in ratings:
            category = getattr(rating, "category", None)
            category_name = category.value if category is not None else "unknown"
            category_counts[category_name] = category_counts.get(category_name, 0) + 1
        summary_parts = [
            f"{category_name}: {count}"
            for category_name, count in sorted(category_counts.items(), key=lambda entry: entry[0])
        ]
        return ", ".join(summary_parts)

    def _pick_feature_row_for_debug(self, feature_rows: list[MessageFeatures]) -> dict[str, Any] | None:
        """
        Pick the most relevant feature row for inspection.

        Prefers runtime rows (`extraction_run_id=None`) because those are the exact
        rows written by live moderation, then falls back to newest extraction row.

        Args:
            feature_rows: Stored feature rows for one flagged message

        Returns:
            Serialized feature-row payload or None when no rows exist
        """
        if not feature_rows:
            return None

        def _sort_key(row: MessageFeatures) -> datetime:
            """
            Normalize feature-row timestamps for deterministic ordering.

            Args:
                row: Feature row being compared

            Returns:
                A datetime usable as a stable max-sort key
            """
            created_at = getattr(row, "created_at", None)
            if isinstance(created_at, datetime):
                return created_at
            return datetime.min.replace(tzinfo=timezone.utc)

        runtime_rows = [row for row in feature_rows if row.extraction_run_id is None]
        selected_row = (
            max(runtime_rows, key=_sort_key)
            if runtime_rows
            else max(feature_rows, key=_sort_key)
        )
        return {
            "extraction_run_id": selected_row.extraction_run_id,
            "run_index": selected_row.run_index,
            "features": selected_row.features or {},
            "created_at": selected_row.created_at,
        }

    def _get_flagged_message_debug_payload(
        self,
        guild_id: int,
        message_id: int,
    ) -> dict[str, Any] | None:
        """
        Fetch all debugging data required to drill into a flagged message.

        Args:
            guild_id: Guild ID used to guard against cross-server lookups
            message_id: Discord message ID of the flagged record

        Returns:
            Serialized debug payload or None if no matching record exists
        """
        session = self.get_db_session()
        try:
            flagged_row = (
                session.query(FlaggedMessage)
                .filter(
                    FlaggedMessage.guild_id == guild_id,
                    FlaggedMessage.message_id == message_id,
                )
                .first()
            )
            if flagged_row is None:
                return None

            feature_rows = (
                session.query(MessageFeatures)
                .filter(MessageFeatures.message_id == message_id)
                .order_by(MessageFeatures.created_at.desc(), MessageFeatures.run_index.desc())
                .all()
            )
            ratings = (
                session.query(FlaggedMessageRating)
                .filter(FlaggedMessageRating.flagged_message_id == message_id)
                .order_by(FlaggedMessageRating.completed_at.desc())
                .all()
            )
        finally:
            session.close()

        target_display = (
            f"<@{flagged_row.target_user_id}>"
            if flagged_row.target_user_id
            else (flagged_row.target_username or "Unknown / None")
        )
        selected_feature_row = self._pick_feature_row_for_debug(feature_rows)
        serialized_context = flagged_row.context_messages or []
        return {
            "message_id": flagged_row.message_id,
            "channel_id": flagged_row.channel_id,
            "author_id": flagged_row.author_id,
            "content": flagged_row.content or "",
            "flagged_at": flagged_row.flagged_at,
            "was_acted_upon": bool(flagged_row.was_acted_upon),
            "waiver_filtered": bool(flagged_row.waiver_filtered),
            "target_display": target_display,
            "feature_row_count": len(feature_rows),
            "selected_feature_row": selected_feature_row,
            "ratings_summary": self._summarize_ratings_for_flagged_message(ratings),
            "context_messages": serialized_context,
            "context_count": len(serialized_context),
        }

    def _get_flagged_message_counts(self, guild_id: int, days: int) -> tuple[int, int]:
        """
        Count recently flagged messages and waiver-filtered messages.

        Args:
            guild_id: Guild ID to scope analytics to the active server
            days: Number of trailing days to include in the analytics window

        Returns:
            Tuple of (total_flagged_count, waiver_filtered_count)
        """
        # Build an absolute UTC cutoff so time-window logic is consistent
        cutoff_timestamp = datetime.now(timezone.utc) - timedelta(days=days)

        session = self.get_db_session()
        try:
            # Apply one shared time filter so both counts use the same window
            base_query = session.query(FlaggedMessage).filter(
                FlaggedMessage.guild_id == guild_id,
                FlaggedMessage.flagged_at >= cutoff_timestamp
            )

            # Count all flagged messages in the requested timeframe
            total_flagged_count = base_query.count()

            # Count only records marked as waiver-filtered in that same timeframe
            waiver_filtered_count = base_query.filter(
                FlaggedMessage.waiver_filtered.is_(True)
            ).count()
        finally:
            session.close()

        return total_flagged_count, waiver_filtered_count

    def _get_flagged_author_leaderboard(
        self,
        guild_id: int,
        days: int,
        include_waiver_filtered: bool,
        include_all_flagged_messages: bool,
        limit: int = 10,
    ) -> tuple[list[tuple[int, int]], int]:
        """
        Build a leaderboard of authors with the most flagged messages.

        Args:
            guild_id: Guild ID to scope the leaderboard to the active server
            days: Number of trailing days to include in the leaderboard window
            include_waiver_filtered: Whether to include waiver-filtered flag rows
            include_all_flagged_messages: If False, only count flags with confirming ratings
            limit: Maximum number of leaderboard rows to return

        Returns:
            Tuple of (leaderboard_rows, total_filtered_flagged_count) where:
            - leaderboard_rows is a list of (author_id, flagged_count) sorted by count descending
            - total_filtered_flagged_count is the total number of flagged messages after filters
        """
        session = self.get_db_session()
        try:
            # Build an absolute UTC cutoff so timeframe filtering is deterministic
            cutoff_timestamp = datetime.now(timezone.utc) - timedelta(days=days)

            # Start from all flagged messages for this guild so cross-server rows are excluded
            flagged_query = session.query(FlaggedMessage).filter(
                FlaggedMessage.guild_id == guild_id,
                FlaggedMessage.flagged_at >= cutoff_timestamp,
            )

            # Optionally remove waiver-filtered rows when building the ranking
            if not include_waiver_filtered:
                flagged_query = flagged_query.filter(FlaggedMessage.waiver_filtered.is_(False))

            # Optionally keep only rows that have at least one rating confirming the flag
            if not include_all_flagged_messages:
                confirming_rating_message_ids = (
                    session.query(FlaggedMessageRating.flagged_message_id)
                    .filter(
                        FlaggedMessageRating.category.in_(
                            [RatingCategory.UNCONSTRUCTIVE, RatingCategory.UNSOLICITED]
                        )
                    )
                    .distinct()
                    .subquery()
                )
                flagged_query = flagged_query.filter(
                    FlaggedMessage.message_id.in_(
                        session.query(confirming_rating_message_ids.c.flagged_message_id)
                    )
                )

            # Capture the full filtered denominator for percentage rendering in the command output
            total_filtered_flagged_count = flagged_query.count()

            # Aggregate by author to rank the members most frequently flagged
            leaderboard_rows = (
                flagged_query.with_entities(
                    FlaggedMessage.author_id,
                    func.count(FlaggedMessage.id).label("flagged_count"),
                )
                .group_by(FlaggedMessage.author_id)
                .order_by(func.count(FlaggedMessage.id).desc())
                .limit(limit)
                .all()
            )
        finally:
            session.close()

        return (
            [(author_id, flagged_count) for author_id, flagged_count in leaderboard_rows],
            total_filtered_flagged_count,
        )

    def _get_recent_flagged_messages(
        self,
        guild_id: int,
        limit: int,
        include_waiver_filtered: bool,
    ) -> list[tuple[int, int, int, datetime, bool]]:
        """
        Fetch the most recent flagged messages for a single guild.

        Args:
            guild_id: Guild ID used to scope results to the current server
            limit: Maximum number of flagged message rows to return
            include_waiver_filtered: Whether rows marked waiver-filtered should be included

        Returns:
            List of tuples containing (message_id, channel_id, author_id, flagged_at, waiver_filtered)
            sorted by newest first
        """
        session = self.get_db_session()
        try:
            # Query only the fields needed for the moderator list to keep memory usage low
            recent_query = (
                session.query(
                    FlaggedMessage.message_id,
                    FlaggedMessage.channel_id,
                    FlaggedMessage.author_id,
                    FlaggedMessage.flagged_at,
                    FlaggedMessage.waiver_filtered,
                )
                .filter(FlaggedMessage.guild_id == guild_id)
            )

            # Allow moderators to include or exclude waiver-filtered rows on demand
            if not include_waiver_filtered:
                recent_query = recent_query.filter(FlaggedMessage.waiver_filtered.is_(False))

            recent_rows = recent_query.order_by(FlaggedMessage.flagged_at.desc()).limit(limit).all()
        finally:
            session.close()

        return [
            (message_id, channel_id, author_id, flagged_at, waiver_filtered)
            for message_id, channel_id, author_id, flagged_at, waiver_filtered in recent_rows
        ]

    @discord.slash_command(name="mod_ping", description="Moderator-only ping command")
    @is_moderator()
    async def mod_ping(self, ctx: discord.ApplicationContext):
        """A moderator-only ping command for testing permissions."""
        await ctx.respond("Moderator ping received!", ephemeral=True)

    @discord.slash_command(name="mod_info", description="Get moderator information")
    @is_moderator()
    async def mod_info(self, ctx: discord.ApplicationContext):
        """Displays moderator-specific information."""
        embed = discord.Embed(
            title="Moderator Info",
            description="Moderator-only bot information.",
            color=discord.Color.red()
        )
        embed.add_field(name="Moderator Roles", value=", ".join(MODERATOR_ROLES), inline=False)
        if isinstance(ctx.author, discord.Member):
            embed.add_field(name="Your Roles", value=", ".join([r.name for r in ctx.author.roles if r.name != "@everyone"]), inline=False)
        
        await ctx.respond(embed=embed, ephemeral=True)

    @discord.slash_command(
        name="mod_flag_analytics",
        description="View flagged message counts for a timeframe",
    )
    @is_moderator()
    async def mod_flag_analytics(
        self,
        ctx: discord.ApplicationContext,
        days: int = 1,
    ):
        """
        Show flagged message analytics for the last N days.

        Args:
            ctx: Discord application context for the command
            days: Number of trailing days to include in the report
        """
        # Guard against invalid or extreme values that don't make operational sense
        if days < 1:
            await ctx.respond(
                "Please provide a timeframe of at least 1 day.",
                ephemeral=True,
            )
            return

        if ctx.guild is None:
            await ctx.respond(
                "This command can only be used in a server.",
                ephemeral=True,
            )
            return

        # Query aggregate counts for the same timeframe window
        total_flagged_count, waiver_filtered_count = self._get_flagged_message_counts(
            guild_id=ctx.guild.id,
            days=days,
        )
        waiver_percentage = (
            (waiver_filtered_count / total_flagged_count) * 100
            if total_flagged_count
            else 0.0
        )

        # Present the analytics in an embed for readability
        embed = discord.Embed(
            title="Flagged Message Analytics",
            color=discord.Color.orange(),
        )
        embed.add_field(
            name="Timeframe",
            value=f"Last {days} day{'s' if days != 1 else ''}",
            inline=False,
        )
        embed.add_field(
            name="Total Flagged Messages",
            value=str(total_flagged_count),
            inline=True,
        )
        embed.add_field(
            name="Waiver-Filtered Flags",
            value=f"{waiver_filtered_count} ({waiver_percentage:.1f}%)",
            inline=True,
        )

        await ctx.respond(embed=embed, ephemeral=True)

    @discord.slash_command(
        name="mod_flag_leaderboard",
        description="Show top members most frequently flagged by moderation",
    )
    @is_moderator()
    async def mod_flag_leaderboard(
        self,
        ctx: discord.ApplicationContext,
        days: int = 30,
        include_waiver_filtered: bool = True,
        include_all_flagged_messages: bool = True,
    ):
        """
        Show the top 10 members most likely to be flagged by the bot.

        Args:
            ctx: Discord application context for the command
            days: Number of trailing days to include in the leaderboard
            include_waiver_filtered: Whether to include waiver-filtered flagged messages
            include_all_flagged_messages: Whether to include all flagged rows instead of only confirmed ones
        """
        # Ensure timeframe input is valid before running DB queries
        if days < 1:
            await ctx.respond(
                "Please provide a timeframe of at least 1 day.",
                ephemeral=True,
            )
            return

        if ctx.guild is None:
            await ctx.respond(
                "This command can only be used in a server.",
                ephemeral=True,
            )
            return

        # Pull a wider pool first, then keep the top ten current guild members
        leaderboard_rows, total_filtered_flagged_count = self._get_flagged_author_leaderboard(
            guild_id=ctx.guild.id,
            days=days,
            include_waiver_filtered=include_waiver_filtered,
            include_all_flagged_messages=include_all_flagged_messages,
            limit=100,
        )

        if not leaderboard_rows:
            await ctx.respond(
                "No flagged message data matched the selected filters.",
                ephemeral=True,
            )
            return

        # Keep leaderboard scoped to current server members as requested
        top_member_rows = [
            (author_id, flagged_count)
            for author_id, flagged_count in leaderboard_rows
            if ctx.guild.get_member(author_id) is not None
        ][:10]

        if not top_member_rows:
            await ctx.respond(
                "No current server members matched the selected filters.",
                ephemeral=True,
            )
            return

        # Render a compact top-10 leaderboard with mentions for easy moderation follow-up
        leaderboard_lines = [
            (
                f"{rank}. <@{author_id}> - {flagged_count} flagged "
                f"({(flagged_count / total_filtered_flagged_count) * 100:.1f}%)"
            )
            for rank, (author_id, flagged_count) in enumerate(top_member_rows, start=1)
        ]

        embed = discord.Embed(
            title="Flag Likelihood Leaderboard",
            description="\n".join(leaderboard_lines),
            color=discord.Color.gold(),
        )
        embed.add_field(
            name="Timeframe",
            value=f"Last {days} day{'s' if days != 1 else ''}",
            inline=True,
        )
        embed.add_field(
            name="Include Waiver-Filtered",
            value="Yes" if include_waiver_filtered else "No",
            inline=True,
        )
        embed.add_field(
            name="Include All Flagged",
            value="Yes" if include_all_flagged_messages else "No (Confirmed only)",
            inline=True,
        )
        if not include_all_flagged_messages:
            embed.add_field(
                name="Confirmed Rating Definition",
                value="Rated as `unconstructive` or `unsolicited`",
                inline=False,
            )

        await ctx.respond(embed=embed, ephemeral=True)

    @discord.slash_command(
        name="mod_flag_recent",
        description="Show the most recent flagged messages with jump links",
    )
    @is_moderator()
    async def mod_flag_recent(
        self,
        ctx: discord.ApplicationContext,
        count: int = 10,
        include_waiver_filtered: bool = True,
    ):
        """
        Show the most recent flagged messages for the current guild.

        Args:
            ctx: Discord application context for the command
            count: Number of recent flagged messages to show
            include_waiver_filtered: Whether to include waiver-filtered flagged messages
        """
        # Keep the response readable and under Discord embed limits
        if count < 1 or count > 25:
            await ctx.respond(
                "Please provide a count between 1 and 25.",
                ephemeral=True,
            )
            return

        if ctx.guild is None:
            await ctx.respond(
                "This command can only be used in a server.",
                ephemeral=True,
            )
            return

        recent_rows = self._get_recent_flagged_messages(
            guild_id=ctx.guild.id,
            limit=count,
            include_waiver_filtered=include_waiver_filtered,
        )
        if not recent_rows:
            await ctx.respond(
                "No flagged messages found for this server.",
                ephemeral=True,
            )
            return

        # Build one concise line per flagged message with direct jump URLs
        recent_lines = [
            (
                f"{index}. <@{author_id}> in <#{channel_id}> at "
                f"<t:{int(flagged_at.timestamp())}:R> - "
                f"[Jump](https://discord.com/channels/{ctx.guild.id}/{channel_id}/{message_id})"
                f"{' (waiver filtered)' if waiver_filtered else ''}"
            )
            for index, (message_id, channel_id, author_id, flagged_at, waiver_filtered)
            in enumerate(recent_rows, start=1)
        ]

        embed = discord.Embed(
            title=f"Most Recent Flagged Messages ({len(recent_rows)})",
            description="\n".join(recent_lines),
            color=discord.Color.orange(),
        )
        embed.add_field(
            name="Guild Scope",
            value=ctx.guild.name,
            inline=False,
        )
        embed.add_field(
            name="Include Waiver-Filtered",
            value="Yes" if include_waiver_filtered else "No",
            inline=False,
        )
        await ctx.respond(embed=embed, ephemeral=True)

    @discord.slash_command(
        name="mod_flag_investigate",
        description="Interactively inspect recent flagged messages and decision details",
    )
    @is_moderator()
    async def mod_flag_investigate(
        self,
        ctx: discord.ApplicationContext,
        count: int = 10,
        include_waiver_filtered: bool = True,
    ):
        """
        Launch an interactive view for recent flagged-message investigation.

        Args:
            ctx: Discord application context for the command
            count: Number of recent flagged records to load in the investigation panel
            include_waiver_filtered: Whether to include waiver-filtered flagged records
        """
        # Keep dataset small enough for readable embeds and select-option limits
        if count < 1 or count > 25:
            await ctx.respond(
                "Please provide a count between 1 and 25.",
                ephemeral=True,
            )
            return

        if ctx.guild is None:
            await ctx.respond(
                "This command can only be used in a server.",
                ephemeral=True,
            )
            return

        recent_records = self._serialize_recent_flagged_records(
            guild_id=ctx.guild.id,
            limit=count,
            include_waiver_filtered=include_waiver_filtered,
        )
        if not recent_records:
            await ctx.respond(
                "No flagged messages found for this server.",
                ephemeral=True,
            )
            return

        view = ModFlagInvestigationView(
            cog=self,
            requesting_user_id=ctx.author.id,
            guild_id=ctx.guild.id,
            recent_records=recent_records,
            selected_message_id=recent_records[0]["message_id"],
        )
        embed = await view.build_current_embed()
        await ctx.respond(embed=embed, view=view, ephemeral=True)

    @discord.slash_command(
        name="mod_flag_drilldown",
        description="Open feature and context drill-down for a specific flagged message ID",
    )
    @is_moderator()
    async def mod_flag_drilldown(
        self,
        ctx: discord.ApplicationContext,
        message_id: str,
    ):
        """
        Open the investigation view focused on one flagged message.

        Args:
            ctx: Discord application context for the command
            message_id: Discord message ID of a previously flagged message
        """
        if ctx.guild is None:
            await ctx.respond(
                "This command can only be used in a server.",
                ephemeral=True,
            )
            return

        if not message_id.isdigit():
            await ctx.respond(
                "Please provide a valid numeric message ID.",
                ephemeral=True,
            )
            return
        selected_message_id = int(message_id)

        payload = self._get_flagged_message_debug_payload(
            guild_id=ctx.guild.id,
            message_id=selected_message_id,
        )
        if payload is None:
            await ctx.respond(
                "No flagged message with that ID was found in this server.",
                ephemeral=True,
            )
            return

        recent_records = self._serialize_recent_flagged_records(
            guild_id=ctx.guild.id,
            limit=25,
            include_waiver_filtered=True,
        )
        # Ensure the requested record appears in the dropdown list even if it is older than top 25
        if selected_message_id not in {record["message_id"] for record in recent_records}:
            selected_record = {
                "message_id": payload["message_id"],
                "channel_id": payload["channel_id"],
                "author_id": payload["author_id"],
                "flagged_at": payload["flagged_at"],
                "waiver_filtered": payload["waiver_filtered"],
            }

            # Keep Discord select options at or below 25 by replacing the oldest visible row
            if len(recent_records) >= 25:
                recent_records = recent_records[:24] + [selected_record]
            else:
                recent_records.append(selected_record)

        view = ModFlagInvestigationView(
            cog=self,
            requesting_user_id=ctx.author.id,
            guild_id=ctx.guild.id,
            recent_records=recent_records,
            selected_message_id=selected_message_id,
        )
        view.mode = "overview"
        view._sync_button_states()
        embed = await view.build_current_embed()
        await ctx.respond(embed=embed, view=view, ephemeral=True)

    @discord.slash_command(name="check", description="Manually trigger moderation for this channel or thread")
    @is_moderator()
    async def check(self, ctx: discord.ApplicationContext):
        """
        Manually trigger the moderation workflow for the current channel or thread.
        
        Args:
            ctx: Discord application context for the command.
        """
        channel = ctx.channel

        # Validate channel type to ensure we can moderate
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            await ctx.respond(
                "This command can only be used in a text channel or thread.",
                ephemeral=True,
            )
            return

        # Ensure the channel is part of the tracked allow list
        if not is_tracked_channel(channel):
            await ctx.respond(
                "This channel is not tracked by the moderation system.",
                ephemeral=True,
            )
            return

        # Acknowledge quickly so the interaction does not time out
        await ctx.respond(
            f"Running moderation for {channel.mention}...", ephemeral=True
        )

        # Run moderation in the background so the command handler returns quickly
        async def _run_and_report():
            try:
                result: "ModerationResult" = await self.bot.run_moderation_now(channel)
                # Build a concise summary of the moderation pass outcome
                summary_lines = [
                    f"Moderation {'succeeded' if result.success else 'failed'} for {channel.mention}",
                    f"Reason: {result.reason}",
                    f"Candidates considered: {result.candidates_considered}",
                    f"Candidates after filters: {result.candidates_after_filters}",
                    f"Flagged messages: {result.flagged_new_count} new, {result.flagged_existing_count} already flagged",
                ]
                summary = "\n".join(summary_lines)
                try:
                    await ctx.interaction.edit_original_response(content=summary)
                except Exception:
                    await ctx.interaction.followup.send(summary, ephemeral=True)
            except Exception:
                # Attempt to notify about failure without blocking the command
                try:
                    await ctx.interaction.edit_original_response(
                        content="Moderation run failed due to an unexpected error."
                    )
                except Exception:
                    await ctx.interaction.followup.send(
                        "Moderation run failed due to an unexpected error.",
                        ephemeral=True,
                    )

        asyncio.create_task(_run_and_report())

    @discord.slash_command(name="retrain", description="Manually trigger model retraining (admin only)")
    @is_admin()
    async def retrain(self, ctx: discord.ApplicationContext):
        """
        Manually trigger retraining of the moderation classifier model.

        This bypasses the normal rating count threshold and immediately
        starts a retraining run using available features.
        """
        await ctx.respond("Starting model retraining...", ephemeral=True)

        async def _run_retrain():
            try:
                from training import retrain_model

                success = await retrain_model()
                if success:
                    result_msg = "✅ Model retraining completed successfully!"
                else:
                    result_msg = "⚠️ Model retraining returned False (check logs for details)"
            except Exception as e:
                result_msg = f"❌ Model retraining failed: {e}"

            try:
                await ctx.interaction.edit_original_response(content=result_msg)
            except Exception:
                await ctx.interaction.followup.send(result_msg, ephemeral=True)

        asyncio.create_task(_run_retrain())

    async def cog_command_error(self, ctx: discord.ApplicationContext, error: Exception):
        """Handle errors for commands in this cog."""
        if isinstance(error, commands.CheckFailure):
            await ctx.respond(
                "You don't have permission to use this command. "
                f"Required roles: {', '.join(MODERATOR_ROLES)}",
                ephemeral=True
            )
        else:
            raise error


def setup(bot):
    """Called by Pycord to load this cog."""
    bot.add_cog(Restricted(bot))

