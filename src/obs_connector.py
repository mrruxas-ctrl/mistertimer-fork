import asyncio
import logging
from dataclasses import dataclass

from obsws_python import ReqClient

log = logging.getLogger(__name__)


@dataclass
class SceneInfo:
    name: str
    index: int


@dataclass
class SceneItemInfo:
    id: int
    name: str
    type: str
    index: int


@dataclass
class SceneItemTransform:
    pos_x: float = 0.0
    pos_y: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: float = 0.0
    width: float = 0.0
    height: float = 0.0
    source_width: float = 0.0
    source_height: float = 0.0
    crop_left: int = 0
    crop_right: int = 0
    crop_top: int = 0
    crop_bottom: int = 0
    alignment: int = 5


class OBSConnector:
    def __init__(self, host: str = "localhost", port: int = 4455, password: str = ""):
        self.host = host
        self.port = port
        self.password = password
        self._client: ReqClient | None = None
        self._canvas_width = 1920
        self._canvas_height = 1080

    @property
    def connected(self) -> bool:
        return self._client is not None

    @property
    def canvas_width(self) -> int:
        return self._canvas_width

    @property
    def canvas_height(self) -> int:
        return self._canvas_height

    def connect(self) -> bool:
        log.info("Connecting to OBS at %s:%s", self.host, self.port)
        try:
            kwargs = {"host": self.host, "port": self.port}
            if self.password:
                kwargs["password"] = self.password
            self._client = ReqClient(**kwargs)
            settings = self._client.send("GetVideoSettings")
            self._canvas_width = settings.base_width
            self._canvas_height = settings.base_height
            log.info(
                "Connected to OBS — canvas %s×%s",
                self._canvas_width,
                self._canvas_height,
            )
            return True
        except Exception as exc:
            log.warning("Failed to connect to OBS: %s", exc)
            self._client = None
            return False

    def disconnect(self):
        if self._client:
            try:
                self._client.disconnect()
            except Exception:
                log.debug("OBS disconnect failed", exc_info=True)
            self._client = None
            log.info("Disconnected from OBS")

    def get_scenes(self) -> list[SceneInfo]:
        if not self._client:
            return []
        try:
            resp = self._client.send("GetSceneList")
            return [
                SceneInfo(
                    name=scene.get("sceneName", ""),
                    index=scene.get("sceneIndex", 0),
                )
                for scene in resp.scenes
            ]
        except Exception as exc:
            log.error("Failed to get scene list: %s", exc)
            return []

    def get_scene_items(self, scene_name: str) -> list[SceneItemInfo]:
        if not self._client:
            return []
        try:
            resp = self._client.send("GetSceneItemList", {"sceneName": scene_name})
            return [
                SceneItemInfo(
                    id=item["sceneItemId"],
                    name=item.get("sourceName", ""),
                    type=item.get("inputKind", ""),
                    index=item.get("sceneItemIndex", 0),
                )
                for item in resp.scene_items
            ]
        except Exception as exc:
            log.error("Failed to get items for scene '%s': %s", scene_name, exc)
            return []

    def get_scene_item_id_by_name(self, scene_name: str, source_name: str) -> int | None:
        for item in self.get_scene_items(scene_name):
            if item.name == source_name:
                return item.id
        return None

    def get_scene_item_transform(
        self, scene_name: str, item_id: int
    ) -> SceneItemTransform | None:
        if not self._client:
            return None
        try:
            resp = self._client.send(
                "GetSceneItemTransform",
                {"sceneName": scene_name, "sceneItemId": item_id},
            )
            t = resp.scene_item_transform
            return SceneItemTransform(
                pos_x=t.get("positionX", 0.0),
                pos_y=t.get("positionY", 0.0),
                scale_x=t.get("scaleX", 1.0),
                scale_y=t.get("scaleY", 1.0),
                rotation=t.get("rotation", 0.0),
                width=t.get("width", 0.0),
                height=t.get("height", 0.0),
                source_width=t.get("sourceWidth", 0.0),
                source_height=t.get("sourceHeight", 0.0),
                crop_left=t.get("cropLeft", 0),
                crop_right=t.get("cropRight", 0),
                crop_top=t.get("cropTop", 0),
                crop_bottom=t.get("cropBottom", 0),
                alignment=t.get("alignment", 5),
            )
        except Exception as exc:
            log.error(
                "Failed to get transform for item %s in scene '%s': %s",
                item_id,
                scene_name,
                exc,
            )
            return None

    def set_scene_item_transform(
        self,
        scene_name: str,
        item_id: int,
        pos_x: float,
        pos_y: float,
        rotation: float = 0.0,
        scale_x: float | None = None,
        scale_y: float | None = None,
        base_transform: SceneItemTransform | None = None,
    ) -> bool:
        if not self._client:
            return False

        current = base_transform or self.get_scene_item_transform(scene_name, item_id)
        if current is None:
            return False

        transform_data = {
            "positionX": float(pos_x),
            "positionY": float(pos_y),
            "rotation": float(rotation),
            "scaleX": float(current.scale_x if scale_x is None else scale_x),
            "scaleY": float(current.scale_y if scale_y is None else scale_y),
            "alignment": int(current.alignment),
            "cropLeft": int(current.crop_left),
            "cropRight": int(current.crop_right),
            "cropTop": int(current.crop_top),
            "cropBottom": int(current.crop_bottom),
        }

        try:
            self._client.send(
                "SetSceneItemTransform",
                {
                    "sceneName": scene_name,
                    "sceneItemId": item_id,
                    "sceneItemTransform": transform_data,
                },
            )
            return True
        except Exception as exc:
            log.error(
                "Failed to set transform for item %s in scene '%s': %s",
                item_id,
                scene_name,
                exc,
            )
            return False

    async def _run_sync(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def async_get_scenes(self) -> list[SceneInfo]:
        return await self._run_sync(self.get_scenes)

    async def async_get_scene_items(self, scene_name: str) -> list[SceneItemInfo]:
        return await self._run_sync(self.get_scene_items, scene_name)

    async def async_get_scene_item_transform(
        self, scene_name: str, item_id: int
    ) -> SceneItemTransform | None:
        return await self._run_sync(self.get_scene_item_transform, scene_name, item_id)

    async def async_connect(self) -> bool:
        return await self._run_sync(self.connect)

    def get_source_screenshot(
        self,
        source_name: str,
        width: int = 640,
        height: int = 360,
        quality: int = 80,
    ) -> str | None:
        if not self._client:
            log.debug("get_source_screenshot: not connected to OBS")
            return None
        try:
            resp = self._client.send(
                "GetSourceScreenshot",
                {
                    "sourceName": source_name,
                    "imageFormat": "jpg",
                    "imageWidth": width,
                    "imageHeight": height,
                    "imageCompressionQuality": quality,
                },
            )
            return resp.image_data
        except Exception as exc:
            log.error("Failed to get screenshot for source '%s': %s", source_name, exc)
            return None
