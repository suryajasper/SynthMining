import m from 'mithril';
import '../css/popup.scss';

const popupStatus = {
  active: true,
}

function updatePopupOverlayStatus({active}) {
  popupStatus.active = active;
}

window.onkeydown = e => {
  if (e.key === 'Escape') {
    updatePopupOverlayStatus({active: false});
    m.redraw();
  }
}

function PopupOverlay() {
  let prevActive = popupStatus.active;

  return {
    view(vnode) {
      if (prevActive !== popupStatus.active && !popupStatus.active)
        vnode.attrs.disabledCallback();
      prevActive = popupStatus.active;

      return m('div.popup-background-blur', {
        class: popupStatus.active ? 'enabled' : '',
      },
        vnode.children, 
      );
    }
  };
}

const PopupOuter = {
  createCallbackButton(action) {
    return m('button.popup-button', {
      onclick: e => {
        action.res(e);
      }
    },
      action.name, 
    )
  },

  view(vnode) {
    return m('div.popup-container', {
      style: { display: vnode.attrs.active ? 'block' : 'none', }
    }, [
      m('div.popup-header', [
        m('span.popup-title', vnode.attrs.title),
      ]),
      m('div.popup-content', 
        vnode.children,
      ),
      m('div.popup-footer', 
        (vnode.attrs.actions || [])
          .map(this.createCallbackButton)
      )
    ]);
  }
}

class Popup {
  constructor() {
    this.input = {};
  }

  createInputGroup({ 
    id, 
    displayTitle=id, 
    type='text', 
    placeholder='' 
  }) {
    return m('div.input-group', [
      m('label.input-group-label', { for: id }, displayTitle),

      type === 'textarea' ?
        m('textarea.input-group-textarea', {
          name: id, placeholder,
        }) :
        m('input.input-group-input', {
          name: id, type, placeholder,
          oninput: e => {
            this.input[id] = e.target.value;
          }
        })
    ]);
  }
}

export { updatePopupOverlayStatus, PopupOverlay, PopupOuter, Popup };