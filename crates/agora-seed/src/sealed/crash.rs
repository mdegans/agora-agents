//! Crash points for tests: an operator command stops at a named step as
//! if the process had died there, so a test can restart and check that
//! nothing was lost. A no-op outside tests.

/// Stop here if a test asked to.
#[cfg(not(test))]
pub fn point(_name: &'static str) -> anyhow::Result<()> {
    Ok(())
}

#[cfg(test)]
thread_local! {
    static AT: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
type Pause = (
    &'static str,
    std::sync::mpsc::Sender<()>,
    std::sync::mpsc::Receiver<()>,
);

#[cfg(test)]
thread_local! {
    static PAUSE: std::cell::RefCell<Option<Pause>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub fn point(name: &'static str) -> anyhow::Result<()> {
    let pause = PAUSE.with(|p| {
        let mut p = p.borrow_mut();
        if p.as_ref().is_some_and(|(at, _, _)| *at == name) {
            p.take()
        } else {
            None
        }
    });
    if let Some((_, reached, release)) = pause {
        // The test runs the other side of the interleaving now
        let _ = reached.send(());
        let _ = release.recv();
    }
    if AT.with(|a| a.get()) == Some(name) {
        AT.with(|a| a.set(None));
        anyhow::bail!("simulated crash at {name}");
    }
    Ok(())
}

/// Crash at `name`, once, on this thread.
#[cfg(test)]
pub fn at(name: &'static str) {
    AT.with(|a| a.set(Some(name)));
}

/// Pause at `name`, once, on this thread: `reached` is sent to when it is
/// reached, and the step waits for `release` (a test's interleaving).
#[cfg(test)]
pub fn pause_at(
    name: &'static str,
    reached: std::sync::mpsc::Sender<()>,
    release: std::sync::mpsc::Receiver<()>,
) {
    PAUSE.with(|p| *p.borrow_mut() = Some((name, reached, release)));
}
