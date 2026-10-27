//! `agora-operator yubikey …`: a YubiKey PIV slot as the operator
//! credential (agora `docs/design-operator-api.md`, "Hardware tiers").
//!
//! - `info`: the device's serial, firmware, PIN state and slots;
//! - `provision`: generate the unattended key (slot 9a, PIN once, touch
//!   never) and the sudo key (9c, PIN once, touch always), and print what
//!   the portal's "Add signing key" form takes for the first;
//! - `attest`: print that again for any slot: the public key, the
//!   attestation (one line of JSON) and the proof of possession;
//! - `reset`: reset the PIV application alone, after a typed confirmation.
//!
//! The key is reached only through agentkit's `yubikey` module (the
//! `yubikey` crate over PC/SC), never by running `ykman`, whose PIN and
//! management key arguments show in `ps`. The PIN is read with
//! agora-agent-lib's `read_secret` (a prompt with echo off, or
//! `--pin-stdin`), or from a file this user alone can read (`--pin-file`,
//! which unattended signing needs); never from argv or the environment.

use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

use agora_agent_lib::secret_input::{SecretSource, read_secret};
use agora_agentkit::ids::OperatorId;
use agora_agentkit::operator_signer::FirmwareVersion;
use agora_agentkit::operator_signer::Pin;
use agora_agentkit::operator_signer::yubikey::{
    self, DeviceInfo, ManagementKey, YubikeySigner, device_info,
};
use agora_agentkit::operator_signer::{
    HardwareKey, KeyPolicy, OperatorSigner, PinPolicy, PivSerial, PivSlot, SignatureAlgorithm,
    TouchPolicy,
};
use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand, ValueEnum};

/// The first firmware whose slots say what they hold (PIV metadata)
const METADATA_MIN_FIRMWARE: FirmwareVersion = FirmwareVersion::new(5, 3, 0);
use url::Url;

use crate::config::{self, OperatorConfig, SignerConfig};

#[derive(Debug, Subcommand)]
pub enum YubikeyCommand {
    /// The connected YubiKey: serial, firmware, PIN state, and each slot's
    /// key and policies. Needs no PIN
    Info {
        #[command(flatten)]
        device: DeviceArgs,
    },

    /// Generate the operator keys on the YubiKey: the unattended key (slot
    /// 9a: PIN once, touch never) and the sudo key (9c: PIN once, touch
    /// always), and print what the portal needs for the first
    Provision(ProvisionArgs),

    /// Print a slot's public key, attestation and proof of possession for
    /// the portal's "Add signing key" form
    Attest(AttestArgs),

    /// Reset the YubiKey's PIV application (and nothing else on it): every
    /// PIV key is destroyed and the PIN, PUK and management key return to
    /// the factory defaults. Asks you to type the serial
    Reset {
        /// The device's serial (`yubikey info` shows it). Required: a reset
        /// never picks "the connected one"
        #[arg(long)]
        serial: u32,
    },
}

/// Where the PIN comes from: by default a prompt on the terminal with echo
/// off (agora-agent-lib's `read_secret`), or one line on stdin, or a file
/// this user alone can read. Never argv or the environment
#[derive(Debug, Args)]
pub struct PinArgs {
    /// Read the PIN from this file (mode 600). Unattended signing needs
    /// one: `provision` records it in the config
    #[arg(long, conflicts_with = "pin_stdin")]
    pub pin_file: Option<PathBuf>,
    /// Read the PIN as one line on stdin, for scripts
    #[arg(long)]
    pub pin_stdin: bool,
}

impl PinArgs {
    /// The PIN, from wherever these arguments say
    pub fn read(&self) -> Result<Pin> {
        if let Some(path) = &self.pin_file {
            return config::read_pin(path);
        }
        let source = match self.pin_stdin {
            true => SecretSource::Stdin,
            false => SecretSource::Prompt,
        };
        let mut pin = read_secret("The YubiKey's PIV PIN", source)?;
        // Moved, not copied: the zeroizing wrapper is left empty
        Ok(Pin::from_secret(agora_agentkit::secrets::Secret::new(
            std::mem::take(&mut *pin),
        )))
    }
}

#[derive(Debug, Args)]
pub struct DeviceArgs {
    /// The YubiKey's serial, when more than one is connected
    #[arg(long)]
    pub serial: Option<u32>,
}

impl DeviceArgs {
    fn serial(&self) -> Option<PivSerial> {
        self.serial.map(PivSerial::new)
    }
}

/// The key algorithm to generate
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Algorithm {
    /// ECDSA P-256: every YubiKey 5
    P256,
    /// Ed25519: firmware 5.7 and later
    Ed25519,
}

impl From<Algorithm> for SignatureAlgorithm {
    fn from(a: Algorithm) -> Self {
        match a {
            Algorithm::P256 => SignatureAlgorithm::EcdsaP256,
            Algorithm::Ed25519 => SignatureAlgorithm::Ed25519,
        }
    }
}

#[derive(Debug, Args)]
pub struct ProvisionArgs {
    #[command(flatten)]
    pub device: DeviceArgs,
    #[command(flatten)]
    pub pin: PinArgs,
    /// The unattended key's slot
    #[arg(long, default_value = "9a", value_parser = parse_slot)]
    pub slot: PivSlot,
    /// The sudo key's slot (touch always)
    #[arg(long, default_value = "9c", value_parser = parse_slot)]
    pub sudo_slot: PivSlot,
    /// Generate the unattended key only
    #[arg(long)]
    pub no_sudo: bool,
    /// The keys' algorithm
    #[arg(long, value_enum, default_value_t = Algorithm::P256)]
    pub algorithm: Algorithm,
    /// The management key, as hex in a file (mode 600). By default the
    /// factory key if the device still has it, else the PIN-protected one
    /// stored on the device, unlocked with the PIN
    #[arg(long)]
    pub management_key_file: Option<PathBuf>,
    /// Replace keys already in the slots (asks you to type the slots)
    #[arg(long)]
    pub overwrite: bool,
    /// Your operator id, as the portal's "Add signing key" page shows it:
    /// also print the proof of possession
    #[arg(long)]
    pub operator: Option<OperatorId>,
    /// The label to add the key under in the portal
    #[arg(long, default_value = "yubikey")]
    pub label: String,
    /// Don't ask before generating into empty slots
    #[arg(long, short)]
    pub yes: bool,
}

#[derive(Debug, Args)]
pub struct AttestArgs {
    #[command(flatten)]
    pub device: DeviceArgs,
    #[command(flatten)]
    pub pin: PinArgs,
    /// The slot
    #[arg(long, default_value = "9a", value_parser = parse_slot)]
    pub slot: PivSlot,
    /// Your operator id, as the portal's "Add signing key" page shows it
    #[arg(long)]
    pub operator: OperatorId,
    /// The label to add the key under in the portal
    #[arg(long, default_value = "yubikey")]
    pub label: String,
}

/// A PIV slot: `9a`, `9c`, `9d`, `9e`, or its name
pub fn parse_slot(text: &str) -> Result<PivSlot, String> {
    let text = text.trim().to_ascii_lowercase();
    PivSlot::ALL
        .into_iter()
        .find(|slot| {
            slot.to_string() == text
                || serde_json::to_value(slot)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_owned))
                    == Some(text.clone())
        })
        .ok_or_else(|| format!("`{text}` is not a PIV key slot: 9a, 9c, 9d or 9e"))
}

pub async fn run(
    command: YubikeyCommand,
    config_path: &Path,
    config: &mut OperatorConfig,
    server: Option<&Url>,
) -> Result<()> {
    match command {
        YubikeyCommand::Info { device } => {
            let info = device_info(device.serial()).await?;
            print!("{}", describe_device(&info));
            Ok(())
        }
        YubikeyCommand::Provision(args) => provision(args, config_path, config, server).await,
        YubikeyCommand::Attest(args) => {
            let pin = args.pin.read()?;
            let signer = YubikeySigner::open_with_pin(args.device.serial(), args.slot, pin)
                .await
                .with_context(|| format!("opening slot {}", args.slot))?;
            print!(
                "{}",
                portal_block(&signer, args.operator, &args.label, page(server, config)).await?
            );
            Ok(())
        }
        YubikeyCommand::Reset { serial } => reset(PivSerial::new(serial)).await,
    }
}

async fn provision(
    args: ProvisionArgs,
    config_path: &Path,
    config: &mut OperatorConfig,
    server: Option<&Url>,
) -> Result<()> {
    let pin_file = args
        .pin
        .pin_file
        .as_deref()
        .map(std::path::absolute)
        .transpose()?;
    // Read now: a bad PIN source fails before the device is touched
    let pin = args.pin.read()?;
    let management_key_file = args
        .management_key_file
        .as_deref()
        .map(std::path::absolute)
        .transpose()?;
    if let Some(file) = &management_key_file {
        config::check_private(file)?;
    }
    if !args.no_sudo && args.slot == args.sudo_slot {
        bail!("the unattended and sudo keys need two slots");
    }

    let info = device_info(args.device.serial()).await?;
    print!("{}", describe_device(&info));
    if info.pin_is_default == Some(true) {
        eprintln!(
            "Warning: this YubiKey's PIN is still the factory default. Anyone holding it can \
             sign as you: change the PIN (and the PUK) before relying on it."
        );
    }

    let mut plan = vec![(args.slot, KeyPolicy::UNATTENDED, "unattended")];
    if !args.no_sudo {
        plan.push((args.sudo_slot, KeyPolicy::SUDO, "sudo"));
    }
    let occupied: Vec<PivSlot> = plan
        .iter()
        .map(|(slot, ..)| *slot)
        .filter(|slot| {
            info.slots
                .iter()
                .any(|s| s.slot == *slot && (s.key.is_some() || s.generated.is_some()))
        })
        .collect();
    // Before firmware 5.3 there is no metadata: an imported key reads as an
    // empty slot, so "empty" is not known and is never taken on a y/n
    let emptiness_unknown = info.firmware < METADATA_MIN_FIRMWARE;
    if occupied.is_empty() && emptiness_unknown {
        let names = plan
            .iter()
            .map(|(slot, ..)| slot.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        typed_confirmation(
            &format!(
                "YubiKey {} (firmware {}) can't say whether slot(s) {names} hold an imported \
                 key; generating destroys any that do. Type `overwrite {names}` to go on:",
                info.serial, info.firmware
            ),
            &format!("overwrite {names}"),
        )?;
    } else if !occupied.is_empty() {
        let names = occupied
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        if !args.overwrite {
            bail!(
                "slot(s) {names} already hold a key; nothing was changed. Pass --overwrite to \
                 replace them (a credential registered with one stops working)"
            );
        }
        typed_confirmation(
            &format!(
                "This destroys the key(s) in slot(s) {names} of YubiKey {}. Type `overwrite \
                 {names}` to go on:",
                info.serial
            ),
            &format!("overwrite {names}"),
        )?;
    } else if !args.yes
        && !crate::confirm(&format!(
            "Generate {} on YubiKey {}?",
            plan.iter()
                .map(|(slot, policy, role)| format!(
                    "the {role} key in slot {slot} ({})",
                    policy_words(*policy)
                ))
                .collect::<Vec<_>>()
                .join(" and "),
            info.serial
        ))?
    {
        bail!("not confirmed; nothing was changed");
    }

    let management_key = || -> Result<ManagementKey> {
        Ok(match &management_key_file {
            Some(file) => ManagementKey::Hex(
                agora_agentkit::secrets::Secret::from_file(file)
                    .with_context(|| format!("reading {}", file.display()))?,
            ),
            None if info.management_key_is_default == Some(true) => ManagementKey::Default,
            None => ManagementKey::Protected(pin.clone()),
        })
    };
    let serial = Some(info.serial);
    let mut unattended = None;
    for (slot, policy, role) in &plan {
        let signer = yubikey::generate(
            serial,
            *slot,
            args.algorithm.into(),
            *policy,
            management_key()?,
        )
        .await
        .map_err(|e| match e {
            agora_agentkit::operator_signer::SignerError::ManagementKeyRefused => anyhow::anyhow!(
                "{e}: pass the key with --management-key-file, or, if it is PIN-protected on \
                 the device, check the PIN file"
            ),
            e => anyhow::Error::new(e),
        })
        .with_context(|| format!("generating the {role} key in slot {slot}"))?;
        println!(
            "Generated the {role} key in slot {slot}: {}",
            signer.public_key().to_hex()
        );
        if *role == "unattended" {
            unattended = Some(signer);
        }
    }

    match &pin_file {
        // Unattended signing reads the PIN from a file: none was given
        None => println!(
            "To sign with slot {} unattended, the config needs a PIN file (mode 600): set \
             its signer to {{ yubikey = {{ slot = \"{}\", pin_file = \"<path>\", serial = {} }} }} \
             in {}.",
            args.slot,
            serde_json::to_value(args.slot)?
                .as_str()
                .unwrap_or_default(),
            info.serial,
            config_path.display()
        ),
        Some(pin_file) => save_signer(config_path, config, args.slot, pin_file, serial)?,
    }

    let signer = unattended.expect("the unattended key is always in the plan");
    signer.verify_pin(&pin).await?;
    match args.operator {
        Some(operator) => print!(
            "{}",
            portal_block(&signer, operator, &args.label, page(server, config)).await?
        ),
        None => println!(
            "Then `agora-operator yubikey attest --slot {} --operator <your operator id>` \
             prints what the portal's \"Add signing key\" form takes.",
            args.slot
        ),
    }
    Ok(())
}

/// Make the config sign with `slot`, its PIN in `pin_file`, unless it signs
/// with another key already (then say how to switch)
fn save_signer(
    config_path: &Path,
    config: &mut OperatorConfig,
    slot: PivSlot,
    pin_file: &Path,
    serial: Option<PivSerial>,
) -> Result<()> {
    let signer_config = SignerConfig::Yubikey {
        slot,
        pin_file: pin_file.to_owned(),
        serial,
    };
    if config.signer.is_none() {
        config.signer = Some(signer_config);
        config::save(config_path, config)?;
        println!(
            "The config now signs with slot {} ({}).",
            slot,
            config_path.display()
        );
    } else {
        println!(
            "The config signs with another key; to use this one, set its signer to \
             {{ yubikey = {{ slot = \"{}\", pin_file = \"{}\", serial = {} }} }} in {}.",
            serde_json::to_value(slot)?.as_str().unwrap_or_default(),
            pin_file.display(),
            serial.map_or("<serial>".to_owned(), |s| s.to_string()),
            config_path.display()
        );
    }

    Ok(())
}

/// What the portal's "Add signing key" form takes for `signer`: its public
/// key, its attestation as one line of JSON, and its proof of possession
/// for `operator`
async fn portal_block(
    signer: &YubikeySigner,
    operator: OperatorId,
    label: &str,
    page: Option<Url>,
) -> Result<String> {
    let attestation = signer.attestation().await?;
    if signer.policy().touch != TouchPolicy::Never {
        eprintln!("Touch the YubiKey to sign the proof of possession…");
    }
    let proof = agora_agentkit::operator_signer::sign_possession(signer, operator).await?;
    let place = match page {
        Some(page) => format!("Add it at {page}"),
        None => "Add it in the portal: Account, Signing keys".to_owned(),
    };
    Ok(format!(
        "\nYubiKey {}, firmware {}, slot {}: {}, {}\n\n\
         Public key:\n\n    {}\n\n\
         Attestation (paste the whole line):\n\n{}\n\n\
         Proof of possession (for operator {operator}):\n\n    {}\n\n\
         {place} (label: {label}), then save the credential id it shows: \
         `agora-operator config --credential <id>`.\n",
        signer.device_serial(),
        signer.firmware(),
        signer.slot(),
        signer.algorithm(),
        policy_words(signer.policy()),
        signer.public_key().to_hex(),
        serde_json::to_string(&attestation)?,
        proof.to_hex(),
    ))
}

fn page(server: Option<&Url>, config: &OperatorConfig) -> Option<Url> {
    server
        .or(config.server.as_ref())
        .and_then(|s| crate::keygen::page_url(s).ok())
}

fn policy_words(policy: KeyPolicy) -> String {
    let pin = match policy.pin {
        PinPolicy::Never => "no PIN",
        PinPolicy::Once => "PIN once",
        PinPolicy::Always => "PIN every signature",
    };
    let touch = match policy.touch {
        TouchPolicy::Never => "touch never",
        TouchPolicy::Always => "touch always",
        TouchPolicy::Cached => "touch cached",
    };
    format!("{pin}, {touch}")
}

/// What `info` prints: the device's own claims (the server trusts only the
/// attestation's chain)
fn describe_device(info: &DeviceInfo) -> String {
    let yes_no = |v: Option<bool>| match v {
        Some(true) => "yes",
        Some(false) => "no",
        None => "unknown (firmware before 5.3)",
    };
    let mut out = format!(
        "YubiKey {}, firmware {}\n  PIN tries left: {}\n  factory PIN: {}, PUK: {}, management \
         key: {}\n",
        info.serial,
        info.firmware,
        info.pin_tries_left,
        yes_no(info.pin_is_default),
        yes_no(info.puk_is_default),
        yes_no(info.management_key_is_default),
    );
    for slot in &info.slots {
        let what = match (&slot.key, slot.generated) {
            (Some(facts), _) => format!(
                "{}, {}, generated on the device: {}",
                facts.public_key.algorithm(),
                policy_words(facts.policy),
                facts.public_key.to_hex()
            ),
            (None, Some(false)) => "an imported key (can't be attested)".to_owned(),
            (None, Some(true)) => "a key that did not attest".to_owned(),
            (None, None) => "empty".to_owned(),
        };
        out.push_str(&format!("  slot {}: {what}\n", slot.slot));
    }
    out
}

/// Ask on the terminal for `expected`, typed exactly. Refuses without a
/// terminal: nothing destructive runs unattended
fn typed_confirmation(question: &str, expected: &str) -> Result<()> {
    if !std::io::stdin().is_terminal() {
        bail!("{question} Not a terminal to ask on; nothing was changed");
    }
    eprint!("{question} ");
    std::io::stderr().flush()?;
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    if line.trim() != expected {
        bail!("not confirmed; nothing was changed");
    }
    Ok(())
}

async fn reset(serial: PivSerial) -> Result<()> {
    let info = device_info(Some(serial)).await?;
    print!("{}", describe_device(&info));
    typed_confirmation(
        &format!(
            "This blocks the PIN and PUK, then resets the PIV application of YubiKey \
             {serial}: every PIV key and certificate on it is destroyed, and credentials \
             registered with them stop working (revoke them in the portal). FIDO, OpenPGP \
             and OTP are untouched. Type `reset {serial}` to go on:"
        ),
        &format!("reset {serial}"),
    )?;
    yubikey::reset_piv(serial).await?;
    println!(
        "YubiKey {serial}'s PIV application is reset: PIN 123456, PUK 12345678 and the \
         factory management key. Change the PIN and PUK before provisioning."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_parse_by_number_or_name() {
        assert_eq!(parse_slot("9a"), Ok(PivSlot::Authentication));
        assert_eq!(parse_slot("9C"), Ok(PivSlot::Signature));
        assert_eq!(parse_slot("key_management"), Ok(PivSlot::KeyManagement));
        assert_eq!(parse_slot("9e"), Ok(PivSlot::CardAuthentication));
        assert!(parse_slot("f9").is_err());
        assert!(parse_slot("82").is_err());
    }

    #[test]
    fn the_operator_policies_are_the_specs() {
        assert_eq!(
            KeyPolicy::UNATTENDED,
            KeyPolicy {
                pin: PinPolicy::Once,
                touch: TouchPolicy::Never
            }
        );
        assert_eq!(KeyPolicy::SUDO.touch, TouchPolicy::Always);
        assert_eq!(policy_words(KeyPolicy::UNATTENDED), "PIN once, touch never");
    }

    /// No argument anywhere in the CLI is a PIN, PUK or management key:
    /// any id naming one must be a file or stdin switch
    #[test]
    fn no_secret_is_taken_on_the_command_line() {
        use clap::CommandFactory;
        const ALLOWED: [&str; 3] = ["pin_file", "pin_stdin", "management_key_file"];
        fn walk(command: &clap::Command, path: &str, seen: &mut usize) {
            for arg in command.get_arguments() {
                let id = arg.get_id().as_str().to_ascii_lowercase();
                *seen += 1;
                assert!(
                    !(id.contains("pin") || id.contains("puk") || id.contains("management"))
                        || ALLOWED.contains(&id.as_str()),
                    "`{path}` takes `{id}` on argv"
                );
            }
            for sub in command.get_subcommands() {
                walk(sub, &format!("{path} {}", sub.get_name()), seen);
            }
        }
        let mut seen = 0;
        walk(&crate::cli::Cli::command(), "agora-operator", &mut seen);
        assert!(seen > 20, "the walk saw the arguments ({seen})");
    }
}
