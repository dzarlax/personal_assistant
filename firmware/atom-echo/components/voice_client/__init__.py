import esphome.codegen as cg
import esphome.config_validation as cv
from esphome import automation
from esphome.components import microphone, speaker, light
from esphome.const import CONF_ID

DEPENDENCIES = ["microphone", "speaker", "light"]
AUTO_LOAD = ["microphone", "speaker"]

voice_client_ns = cg.esphome_ns.namespace("voice_client")
VoiceClient = voice_client_ns.class_("VoiceClient", cg.Component)

CONF_MICROPHONE = "microphone"
CONF_SPEAKER = "speaker"
CONF_LED = "led"
CONF_API_URL = "api_url"
CONF_API_TOKEN = "api_token"
CONF_MAX_RECORD_SECONDS = "max_record_seconds"

CONFIG_SCHEMA = cv.Schema(
    {
        cv.GenerateID(): cv.declare_id(VoiceClient),
        cv.Required(CONF_MICROPHONE): cv.use_id(microphone.Microphone),
        cv.Required(CONF_SPEAKER): cv.use_id(speaker.Speaker),
        cv.Required(CONF_LED): cv.use_id(light.LightState),
        cv.Required(CONF_API_URL): cv.string,
        cv.Required(CONF_API_TOKEN): cv.string,
        cv.Optional(CONF_MAX_RECORD_SECONDS, default=10): cv.int_range(min=1, max=30),
    }
).extend(cv.COMPONENT_SCHEMA)


async def to_code(config):
    from esphome.components.esp32 import add_idf_component
    add_idf_component(name="espressif/esp_websocket_client", ref="1.2.3")

    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    mic = await cg.get_variable(config[CONF_MICROPHONE])
    cg.add(var.set_microphone(mic))

    spk = await cg.get_variable(config[CONF_SPEAKER])
    cg.add(var.set_speaker(spk))

    led = await cg.get_variable(config[CONF_LED])
    cg.add(var.set_led(led))

    cg.add(var.set_api_url(config[CONF_API_URL]))
    cg.add(var.set_api_token(config[CONF_API_TOKEN]))
    cg.add(var.set_max_record_seconds(config[CONF_MAX_RECORD_SECONDS]))
